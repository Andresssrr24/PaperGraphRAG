# generate Groq responses
import asyncio
import json
from groq import AsyncGroq, Groq

from pydantic import BaseModel
from typing import List

import sys
from pathlib import Path
# Add parent directory to path for global imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_agent_config, get_api_keys

# agent configs
summarization_cfg = get_agent_config("summarization")
query_expansion_cfg = get_agent_config("query_expansion")
api_keys = get_api_keys()

# API key
GROQ_API_KEY = api_keys.get("groq")

# system prompts from config
QUERY_EXPANSION_PROMPT = query_expansion_cfg.get("system_prompt", "")
SUMMARY_PROMPT = summarization_cfg.get("system_prompt", "")


class ExpandedQueries(BaseModel):
    expanded_queries: List[str]


class GroqAsync:
    """Async Groq client for chunk summarization"""
    
    def __init__(self, api_key: str = GROQ_API_KEY):
        self.client = AsyncGroq(api_key=api_key)
        # concurrency settings from config
        max_concurrent = summarization_cfg.get("concurrency", {}).get("max_concurrent", 5)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # model settings from config
        self.model = summarization_cfg.get("model", "openai/gpt-oss-20b")
        self.temperature = summarization_cfg.get("temperature", 0.0)
        self.max_tokens = summarization_cfg.get("max_tokens", 1024)
        self.reasoning_enabled = summarization_cfg.get("reasoning", {}).get("enabled", True)
        self.reasoning_effort = summarization_cfg.get("reasoning", {}).get("effort", "medium")
        self.system_prompt = SUMMARY_PROMPT

    async def summarize_chunk(self, chunk, query):
        """Call LLM to summarize single chunks with its neighbors"""
        async with self.semaphore:
            # format the prompt with the chunk data
            formatted_prompt = self.system_prompt.format(
                query=query,
                text=chunk["text"],
                neighbors="\n".join(chunk.get("neighbors", [])),
            )
            
            kwargs = {
                "model": self.model,
                "max_completion_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "user", "content": formatted_prompt}
                ]
            }
            
            if self.reasoning_enabled:
                kwargs["include_reasoning"] = True
                kwargs["reasoning_effort"] = self.reasoning_effort
            
            rsp = await self.client.chat.completions.create(**kwargs)
        
        return {
            "chunk_id": chunk["chunkId"],
            "summary": rsp.choices[0].message.content    
        }

    async def summarize_all_chunks(self, chunks, query):
        tasks = [self.summarize_chunk(c, query) for c in chunks]
        print(f"\n\n\n\n Number of chunks to summarize: {len(tasks)}")
        summaries = await asyncio.gather(*tasks)
        return summaries


class GroqResponse:
    """Sync Groq client for query expansion and general responses"""
    
    def __init__(self, api_key: str = GROQ_API_KEY):
        self.client = Groq(api_key=api_key)
        
        # model settings from config
        self.model = query_expansion_cfg.get("model", "openai/gpt-oss-20b")
        self.temperature = query_expansion_cfg.get("temperature", 0.0)
        self.max_tokens = query_expansion_cfg.get("max_tokens", 2048)
        self.reasoning_enabled = query_expansion_cfg.get("reasoning", {}).get("enabled", True)
        self.reasoning_effort = query_expansion_cfg.get("reasoning", {}).get("effort", "high")
        self.system_prompt = QUERY_EXPANSION_PROMPT

    def groq_response(self, query):
        """When structured output is not needed"""
        kwargs = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query}
            ]
        }
        
        if self.reasoning_enabled:
            kwargs["include_reasoning"] = True
            kwargs["reasoning_effort"] = self.reasoning_effort
        
        rsp = self.client.chat.completions.create(**kwargs)
        return rsp.choices[0].message.content

    def generate_structured(self, query):
        """Usually called for query expansion - returns structured output"""
        kwargs = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ExpandedQueries",
                    "schema": ExpandedQueries.model_json_schema()}
            }
        }
        
        if self.reasoning_enabled:
            kwargs["include_reasoning"] = True
            kwargs["reasoning_effort"] = self.reasoning_effort
        
        rsp = self.client.chat.completions.create(**kwargs)
        
        ans = ExpandedQueries.model_validate(json.loads(rsp.choices[0].message.content))
        return json.dumps(ans.model_dump(), indent=2)