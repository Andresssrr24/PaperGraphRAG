from openai import OpenAI

import sys
from pathlib import Path
# Add parent directory to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_agent_config, get_api_keys

# agent config
response_cfg = get_agent_config("response_generation")
api_keys = get_api_keys()

OPENAI_API_KEY = api_keys.get("openai")

# model settings
DEFAULT_MODEL = response_cfg.get("model", "gpt-5-nano")
DEFAULT_TEMPERATURE = response_cfg.get("temperature", 0.7)
DEFAULT_MAX_TOKENS = response_cfg.get("max_tokens", 2000)
REASONING_ENABLED = response_cfg.get("reasoning", {}).get("enabled", True)
REASONING_EFFORT = response_cfg.get("reasoning", {}).get("effort", "low")

MAIN_PROMPT = response_cfg.get("system_prompt", "")

class OpenAIResponse:
    def __init__(self, api_key: str = OPENAI_API_KEY):
        self.api_key = api_key

    def call_gpt_5(
        self,
        query: str = None, 
        summary: str = None, 
        metadata: dict = None,
        model_id: str = None,
        api_key: str = None
    ):
        """generate final response using OpenAI model
        Args:
            query: The user's original query
            summary: Summarized context from RAG retrieval
            metadata: Chunk metadata for citations
            model_id: Override default model from config
            api_key: Override default API key from config
        """
        client = OpenAI(api_key=api_key or self.api_key)

        if not summary and not query:
            raise ValueError("No context or query available to generate factual response.")

        # format the prompt with provided data
        prompt = MAIN_PROMPT.format(
            query=query,
            final_summary=summary,
            chunks_metadata=metadata
        )

        kwargs = {
            "model": model_id or DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": DEFAULT_MAX_TOKENS,
            "temperature": DEFAULT_TEMPERATURE,
        }
        
        if REASONING_ENABLED:
            kwargs["reasoning_effort"] = REASONING_EFFORT

        rsp = client.chat.completions.create(**kwargs)

        return rsp.choices[0].message.content.strip()
