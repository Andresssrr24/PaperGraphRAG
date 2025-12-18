from final_knowledge_graph import neo_connection

def check_chunks():
    kg = neo_connection()
    
    query = """
    MATCH (p:Paper)
    OPTIONAL MATCH (p)-[:HAS_CHUNK]->(c:Chunk)
    RETURN p.title AS title, count(c) AS chunk_count, count(c.embedding) AS embedding_count
    """
    
    results = kg.query(query)
    
    print(f"{'Paper Title':<60} | {'Chunks':<10} | {'Embeddings':<10}")
    print("-" * 86)
    for r in results:
        print(f"{r['title']:<60} | {r['chunk_count']:<10} | {r['embedding_count']:<10}")

    # Check for GPT paper specifically
    gpt_query = """
    MATCH (p:Paper {title: 'Improving Language Understanding by Generative Pre-Training'})
    OPTIONAL MATCH (p)-[:HAS_CHUNK]->(c:Chunk)
    RETURN p.title AS title, c.chunkId AS chunkId, size(c.embedding) AS embedding_size
    LIMIT 5
    """
    print("\nGPT Paper Sample Chunks:")
    gpt_results = kg.query(gpt_query)
    for r in gpt_results:
        print(r)

if __name__ == "__main__":
    check_chunks()
