from toon_format import encode, decode
from typing import List, Dict, Any
import asyncio
import json

import os
from dotenv import load_dotenv

from llm.groq_client import GroqResponse, GroqAsync
from llm.openai_client import OpenAIResponse
from knowledge_graph import EMBEDDER # ensure both graph and rag use same embedder
from config import get_api_keys, get_rag_config

load_dotenv()

# init clients
api_keys = get_api_keys()
groq_client = GroqResponse(api_keys.get("groq")) 
groq_async_client = GroqAsync(api_keys.get("groq")) 
openai_client = OpenAIResponse(api_keys.get("openai"))

class GraphRetrieval:
    """Search in text chunks by default"""
    def __init__(
        self,
        kg,
        vector_index: str = "paper_chunks",
        text_index: str = "chunk_text_index",
        embedding_property: str = "embedding",
        id_property: str = "chunkId",
        text_property: str = "text", 
        embedder = EMBEDDER,
    ) -> None:
        self.kg = kg
        self.embedder = embedder
        self.vector_index = vector_index
        self.text_index = text_index
        self.embedding_property = embedding_property
        self.id_property = id_property
        self.text_property = text_property

    def expand_query(self, query: str):
        """Expand query using LLM"""
        return groq_client.generate_structured(query)

    def _embed_query(self):
        try:
            # Openai embeddings
            return self.embedder.embed_query(self.text_query) 
        except Exception:
            raise NotImplementedError("Not implemented yet")

    def _format_metadata_for_llm(self, meta: Dict) -> List[str]:
        """
        Format metadata as concise text blocks for LLM consumption.
        Returns list of formatted strings, one per paper.
        """
        formatted_blocks = []
        
        if not meta:
            return formatted_blocks
            
        for key, data in meta.items():
            if not data.get("paper_title"):
                continue
                
            paper_title = data["paper_title"]
            if isinstance(paper_title, list) and paper_title:
                paper_title = paper_title[0]
            elif not isinstance(paper_title, str):
                continue
                
            # Build concise metadata block
            block_parts = []
            
            # Always include title
            block_parts.append(f"Title: {paper_title}")
            
            # Optional fields - only include if they exist
            if data.get("paper_year"):
                block_parts.append(f"Year: {data['paper_year']}")
                
            if data.get("paper_url"):
                # Extract just the domain/identifier for brevity
                url = data['paper_url']
                if 'arxiv.org' in url:
                    block_parts.append(f"arXiv: {url.split('/')[-1]}")
                else:
                    block_parts.append(f"URL: {url}")
                    
            # Concatenate list fields with ';' instead of separate lines
            if data.get("addressed_task"):
                tasks = data["addressed_task"]
                if isinstance(tasks, list) and tasks:
                    task_str = "; ".join(tasks[:3])  # Limit to top 3
                    block_parts.append(f"Tasks: {task_str}")
                    
            if data.get("datasets_used"):
                datasets = data["datasets_used"]
                if isinstance(datasets, list) and datasets:
                    dataset_str = "; ".join(datasets[:2])  # Limit to top 2
                    block_parts.append(f"Datasets: {dataset_str}")
                    
            if data.get("model"):
                models = data["model"]
                if isinstance(models, list) and models:
                    model_str = "; ".join(models[:2])  # Limit to top 2
                    block_parts.append(f"Models: {model_str}")
                    
            if data.get("score") and isinstance(data["score"], (int, float)):
                block_parts.append(f"Relevance: {data['score']:.2f}")
                
            # Join all parts with ' | ' separator for compactness
            formatted_blocks.append(" | ".join(block_parts))
            
        return formatted_blocks

    def _deduplicate_and_format_metadata(self, all_metadata: Dict) -> str:
        """
        Deduplicate by paper title and return formatted string for LLM.
        """
        seen_titles = set()
        unique_metadata = {}
        
        for key, meta in all_metadata.items():
            paper_title = None
            if isinstance(meta.get("paper_title"), list) and meta["paper_title"]:
                paper_title = meta["paper_title"][0]
            elif isinstance(meta.get("paper_title"), str):
                paper_title = meta["paper_title"]
            
            if paper_title and paper_title not in seen_titles:
                seen_titles.add(paper_title)
                # Keep a simpler version
                unique_metadata[paper_title] = {
                    "paper_title": paper_title,
                    "paper_year": meta.get("paper_year"),
                    "paper_url": meta.get("paper_url"),
                    "addressed_task": meta.get("addressed_task"),
                    "datasets_used": meta.get("datasets_used"),
                    "model": meta.get("model"),
                    "score": meta.get("score"),
                }
        
        # Format for LLM
        formatted_blocks = self._format_metadata_for_llm(
            {f"paper_{i}": meta for i, meta in enumerate(unique_metadata.values())}
        )
        
        # Return as a single string with newline separation
        return "\n".join(formatted_blocks)

    def  vector_search(self, text_query: str, k: int = 10) -> List[Dict]:
        self.text_query = text_query

        emb = self._embed_query()

        cypher = f"""
            WITH $embedding AS queryEmbedding
            CALL db.index.vector.queryNodes (
                '{self.vector_index}',
                $k,
                queryEmbedding
            ) 
            YIELD node, score

            MATCH (p:Paper)-[:HAS_CHUNK]->(node)

            RETURN 
                node.{self.id_property} AS chunkId,
                node.{self.text_property} AS text,
                p.title AS paperTitle,
                score
            ORDER BY score DESC
        """

        return self.kg.query(cypher, params={"k": k, "embedding": emb})

    async def async_vector_search(self, text_query: str, k: int = 3) -> List[Dict]:
        """Wrapper for vector search"""
        return self.vector_search(text_query, k=k)

    def expand_context(self, chunk_id: str, hops: int) -> Dict[str, Any]:
        """Expand retrieval using edges"""
    
        cypher = f"""
        MATCH (c:Chunk {{{self.id_property}: $id}})

        OPTIONAL MATCH p_path = (paper:Paper)-[:HAS_CHUNK*1..{hops}]->(c)

        // Tasks Datasets Models from the paper
        OPTIONAL MATCH (paper)-[:ADDRESSES_TASK]->(t:Task)
        OPTIONAL MATCH (paper)-[:USES_DATASET]->(d:Dataset)
        OPTIONAL MATCH (paper)-[:EXPLAINS_MODEL]->(m:Model)

        // model lineage
        OPTIONAL MATCH lineage = (m)-[:DERIVED_FROM*1..{hops}]->(ancestor:Model)

        WITH
            c,
            paper,

            // collected items
            COLLECT(DISTINCT t.name) AS tasks,
            COLLECT(DISTINCT d.name) AS datasets,
            COLLECT(DISTINCT m.name) AS explainedModels,
            COLLECT(DISTINCT ancestor.name) AS modelLineage,

            // paper titles
            CASE WHEN p_path IS NULL THEN [] ELSE
                [n IN nodes(p_path) | n.title]
            END AS paperTitles

        // chunk neighbors
        CALL (c) {{
            OPTIONAL MATCH (c)-[:SIMILAR_TO*1..{hops}]->(neighbor:Chunk)

            WITH DISTINCT neighbor LIMIT 10
            RETURN COLLECT(neighbor.text) AS neighbors
        }}

        RETURN
            c.{self.id_property} AS chunkId,
            c.{self.text_property} AS text,

            // paper level info
            paper.paperId AS paperId,
            paper.title AS paperTitle,
            paper.year AS paperYear,
            paper.url AS paperUrl,

            paperTitles AS papers,
            tasks,
            datasets,
            explainedModels,
            modelLineage,
            neighbors
        """

        rows = self.kg.query(cypher, params={"id": chunk_id})
        return rows[0] if rows else {}

    async def retrieve(self, query: str, top_k: int = 3, expand: bool = False, hops: int = None):
        """High level RAG: embeds query, vector searches, expands context, returns chunks for LLM"""
        print(f"RAG Query: {query}")

        # first, perform vector search
        print("Performing vector search...")
        results = await self.async_vector_search(query, k=top_k)
        print("Vector search results:", results)

        if not expand:
            return results, []

        if not hops:
            raise ValueError("Specify a value for # of hops in expanded search")
        
        # expand context with graph neighbors
        self.enriched = []
        self.metadata = {} # metadata for final call 
        
        count = 0
        for r in results:
            meta = self.expand_context(r["chunkId"], hops=hops)
            self.enriched.append({
                "chunkId": r["chunkId"],
                "text": r["text"],
                "neighbors": meta.get("neighbors", []),
            })

            self.metadata[f'source_{count}'] = {
                "score": r["score"],
                "paper_title": meta.get("papers", []),
                "paper_url": meta.get("paperUrl", []),
                "paper_year": meta.get("paperYear", []),
                "addressed_task": meta.get("tasks", []),
                "datasets_used": meta.get("datasets", []),
                "model": meta.get("explainedModels", []),
                "model_lineage": meta.get("modelLineage", []),
            }
            count +=1

        print("\n\nExpanded context with graph neighbors:\n", self.enriched)
        return self.enriched, self.metadata

    async def summarize_retrieval(self, query: str, expanded_chunks: list):
        """async RAG context expansion with llm summarization"""
        
        # expanded_chunks is now a flat list of unique chunks
        summaries = await groq_async_client.summarize_all_chunks(expanded_chunks, query)
        print(f"Summarized {len(summaries)} chunks")
            
        self.final_summary = "\n".join(s["summary"] for s in summaries)
        return self.final_summary

    async def generate_response(self, query, top_k=3, expand=True, hops=1):   
        """Generate final response"""
        # expand_query is sync, returns JSON string
        expanded_result = self.expand_query(query)
        expanded_queries = json.loads(expanded_result).get("expanded_queries", [query])
        print(f"Expanded queries: {expanded_queries}")
        # save queries to file
        with open("expanded_queries.json", "w") as f:
            json.dump(expanded_queries, f)

        # parallel vector search
        search_tasks = [self.retrieve(q, top_k=top_k, expand=expand, hops=hops) for q in expanded_queries]
        search_results = await asyncio.gather(*search_tasks)
        
        # search_results is a list of tuples (per expanded query): [(chunks, metadata), (chunks, metadata), ...]
        all_chunks = []
        all_metadata = {}
        
        for i, (chunks, meta) in enumerate(search_results):
            all_chunks.extend(chunks)           
            # Let's try to preserve metadata by using chunkId as key in a temporary dict
            for key, chunk_meta in meta.items():
                #prefix with query index
                all_metadata[f"q{i}_{key}"] = chunk_meta

        # Deduplicate chunks by chunkId
        unique_chunks_map = {c["chunkId"]: c for c in all_chunks}
        unique_chunks = list(unique_chunks_map.values())

        print(f"Retrieved {len(unique_chunks)} unique chunks from {len(all_chunks)} total")

        # Deduplicate metadata by paper_title
        unique_metadata = self._deduplicate_and_format_metadata(all_metadata)

        print(f"Retrieved {len(unique_metadata)} unique metadata from {len(all_metadata)} total")
        
        summary = await self.summarize_retrieval(query=query, expanded_chunks=unique_chunks)
        
        print(f"\n\nQuery: {query}")
        print(f"\n\nSummary: {summary}")
        # print(f"\n\nMetadata: {all_metadata}") # Optional: print full metadata
        
        final_rsp = openai_client.call_gpt_5(query=query, summary=summary, metadata=unique_metadata)

        return final_rsp

'''def hybrid_search(self, query: str, k: int = 10, keyword_weight=0.2):
        """lexical score + vector score"""
        # perform vector search
        vector_results = self.vector_search(query, k=k)

        vec_nodes = {r["id"]: r["score"] for r in vector_results}
        
        cypher = f"""
        UNWIND $vec_nodes AS v
        WITH v.id AS vec_id, v.score AS vector_score

        MATCH (n)
        WHERE n.{self.id_property} = vec_id

        CALL db.index.fulltext.queryNodes(
            '{self.text_index},
            $query
        ) YIELD node AS kw_node, score AS keyword_score

        WHERE kw_nodes.{self.id_property} = vec_id

        RETURN
            kw_node.{self.id_property} AS id,
            kw_node.{self.text_property} AS text,
            vector_score + $keyword_weight * keyword_score AS combined_score
        ORDER BY combined_score DESC
        LIMIT $k
        """

        return self.kg.query(
            cypher,
            params={
                "query": query,
                "vec_nodes": [{'id': nid, 'score': score} for nid, score in vec_nodes.items()],
                "keyword_weight": keyword_weight,
                "k": k,
            }
        )'''

"""rsp, metadata = asyncio.run(rag.retrieve(test_query))
print(rsp)
print(metadata)"""