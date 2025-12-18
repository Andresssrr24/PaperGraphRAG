import os
from dotenv import load_dotenv
import json
import fitz
import hashlib

from langchain_neo4j import Neo4jGraph
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    get_neo4j_config,
    get_embedding_config,
    get_chunking_config,
    get_indexes_config,
    get_similarity_config,
    get_api_keys,
)

# Load configuration
load_dotenv()
neo4j_cfg = get_neo4j_config()
embedding_cfg = get_embedding_config()
chunking_cfg = get_chunking_config()
indexes_cfg = get_indexes_config()
similarity_cfg = get_similarity_config()
api_keys = get_api_keys()

# Neo4j Configuration
NEO4J_URI = neo4j_cfg.get("uri", "neo4j://localhost:7687")
NEO4J_USERNAME = neo4j_cfg.get("username", "neo4j")
NEO4J_PASSWORD = neo4j_cfg.get("password", os.getenv("NEO4J_PASSWORD", ""))
NEO4J_DATABASE = neo4j_cfg.get("database", "neo4j")

# Embedding Configuration
EMBEDDING_MODEL = embedding_cfg.get("model", "text-embedding-3-small")
EMB_DIM = embedding_cfg.get("dimensions", 1536)
OPENAI_API_KEY = api_keys.get("openai") or os.getenv("OPENAI_API_KEY")

EMBEDDER = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model=EMBEDDING_MODEL
)

# Top-k neighbors for similarity relationships
TOP_K = similarity_cfg.get("top_k_neighbors", 2)

# Index names from config
CHUNK_INDEX_NAME = indexes_cfg.get("chunk_vector_index", "paper_chunks")
PAPER_INDEX_NAME = indexes_cfg.get("paper_vector_index", "paper_index")
TEXT_INDEX = indexes_cfg.get("text_fulltext_index", "chunk_text_index")
NODE_FOR_TEXT_INDEX = indexes_cfg.get("node_for_text_index", "Chunk")
PROPERTY_FOR_TEXT_INDEX = indexes_cfg.get("property_for_text_index", "text")

def neo_connection():
    kg = Neo4jGraph(
        url=NEO4J_URI, 
        username=NEO4J_USERNAME, 
        password=NEO4J_PASSWORD, 
        database=NEO4J_DATABASE,
    )
    return kg

kg = neo_connection()

# Chunking configuration
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunking_cfg.get("chunk_size", 1000),
    chunk_overlap=chunking_cfg.get("chunk_overlap", 150),
    is_separator_regex=chunking_cfg.get("is_separator_regex", False)
)    

paper_paths = [
        {"bert": 'knowledge_graphs/data/bert_paper.pdf'},
        {"distilbert": 'knowledge_graphs/data/distilbert_paper.pdf'},
        {"tinybert": 'knowledge_graphs/data/tinybert_paper.pdf'},
        {"mobilebert": 'knowledge_graphs/data/mobilebert_paper.pdf'},
        {"gpt": 'knowledge_graphs/data/gpt_paper.pdf'}
    ]

class PaperNodes:
    def __init__(self) -> None:
        pass

    def _generate_paper_id(self, title: str) -> str:
        return hashlib.md5(title.encode()).hexdigest()[:8]

    def _normalize_paper_paths(self):
        flat = {}
        for entry in self.paper_paths:
            for model_name, path in entry.items():
                flat[model_name] = path
        return flat

    def _match_papers(self):
        paper_paths_map = self._normalize_paper_paths()

        matched = []

        for meta in self.papers_metadata:
            model_name = meta["model"].lower()

            if model_name not in paper_paths_map:
                raise ValueError(f"[ERROR] Missing PDF path for model: {model_name}")

            pdf_path = paper_paths_map[model_name]

            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"[ERROR] File not found: {pdf_path}")

            matched.append({
                "pdf_path": pdf_path,
                "metadata": meta
            })

        return matched

    # build paper level metadata
    def _paper_level_metadata(self) -> list:
        """Build metadata for nodes Paper, Task, Dataset, and Model"""
        self.papers_metadata = []

        for paper in self.json_metadata:
            paper_id = self._generate_paper_id(paper["paper"])
            self.papers_metadata.append({
                "paper_id": paper_id,

                "paper_node": {
                    "title": paper["paper"],
                    "year": paper["year"],
                    "url": paper["url"],
                },

                "tasks": paper["tasks"],
                "datasets": paper["datasets"],
                "model": paper["model"],

                "derived_from": paper["derived_from"]
            })

        return self.papers_metadata

    # calculate paper level embeddings (mean of text chunks embeddings)
    def calculate_paper_level_embeddings(self) -> list:
        papers = self.kg.query("""
            MATCH (p:Paper)-[:HAS_CHUNK]->(c:Chunk)
            RETURN p.paperId as pid, collect(c.embedding) AS embeddings
        """)

        self.paper_embeddings = []
        for row in papers:
            pid = row['pid']
            embs = row['embeddings']

            if not embs:
                continue

            paper_emb = np.mean(np.array(embs), axis=0).tolist()
            self.paper_embeddings.append((pid, paper_emb))
        
        return self.paper_embeddings

    def add_paper_embeddings(self):
        counter = 0
        for pid, emb in self.paper_embeddings:
            self.kg.query(
                """
                MATCH (p:Paper {paperId: $pid})
                SET p.embedding = $emb
                """,

                params={"pid": pid, "emb": emb}
            ) 

            counter +=  1
        print(f"Added embeddings for {counter} papers")
               
 
class ChunkNodes(PaperNodes):
    def __init__(self, k: int = 7) -> None:
        super().__init__()
        self.calculate_embeddings = True
        self.k = k

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text    

    def _chunk_pdf_with_metadata(self, pdf_path, metadata):
        pdf_text = self._extract_text_from_pdf(pdf_path)
        chunks = splitter.split_text(pdf_text)
        print(f"{len(chunks)} total chunks")

        paper_id = metadata["paper_id"]

        chunks_with_metadata = []        
        for i, chunk in enumerate(chunks):
            if self.calculate_embeddings:
                print("Calculating chunk embedding")
                # embed chunk
                emb = EMBEDDER.embed_documents([chunk])[0]

            chunks_with_metadata.append({
                'chunk_id': f"{paper_id}_{i+1}",
                'paper_id': paper_id,
                'text': chunk,
                "embedding": emb if self.calculate_embeddings else [],
                "embedding_model": EMBEDDING_MODEL if self.calculate_embeddings else "",
            })
        
        print(f"Processed {len(chunks_with_metadata)} chunks")
        return chunks_with_metadata

    # build chunk level metadata
    def _chunk_level_metadata(self) -> list:
        matches = self._match_papers()

        self.all_chunks = []
        for entry in matches:
            chunks = self._chunk_pdf_with_metadata(
                pdf_path=entry["pdf_path"],
                metadata=entry["metadata"]
            )

            for chunk in chunks:
                chunk["paper_id"] = entry["metadata"]["paper_id"]

            self.all_chunks.extend(chunks)

        return self.all_chunks

    def _wait_for_index(self, kg, index_name: str, max_retries: int = 30):
        import time
        print(f"Waiting for index {index_name} to be ready...")
        for _ in range(max_retries):
            try:
                res = kg.query(
                    "SHOW INDEXES YIELD name, state, populationPercent WHERE name = $name RETURN state, populationPercent",
                    params={"name": index_name}
                )
                if res:
                    state = res[0]["state"]
                    percent = res[0]["populationPercent"]
                    if state == "ONLINE" and percent >= 100.0:
                        print(f"Index {index_name} is ready (100% populated).")
                        return
            except Exception as e:
                print(f"Error checking index status: {e}")
            
            time.sleep(1)
        print(f"Warning: Index {index_name} timed out waiting for population.")

    def _chunks_knn_relationships(self, kg, k: int = None, batch_size: int = 32):
        print("Building chunk level relationships")
        if k:
            self.k = k

        index_name = CHUNK_INDEX_NAME
        
        # Wait for index to be populated before querying
        self._wait_for_index(kg, index_name)

        rows = kg.query("""
        MATCH (c:Chunk)
        WHERE c.embedding IS NOT NULL AND size(c.embedding) > 0
        RETURN c.chunkId AS id
        """)
        chunk_ids = [r["id"] for r in rows]

        total_created = 0
        # 2) batch over chunk ids to avoid huge transactions
        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i:i+batch_size]
            for cid in batch:
                # call vector query for this chunk and create relationships
                q = f"""
                MATCH (c:Chunk {{chunkId: $cid}})
                CALL db.index.vector.queryNodes('{index_name}', $k, c.embedding) YIELD node, score
                WHERE node.chunkId <> $cid AND node.embedding IS NOT NULL
                MERGE (c)-[r:SIMILAR_TO]->(node)
                SET r.score = score
                RETURN count(r) as created
                """
                res = kg.query(q, params={"cid": cid, "k": self.k})
                # res might be [{'created': X}]
                try:
                    created_for_c = res[0].get("created", 0)
                except Exception:
                    created_for_c = 0
                total_created += created_for_c

        # final check
        cnt = kg.query("MATCH (:Chunk)-[r:SIMILAR_TO]->(:Chunk) RETURN count(r) AS rels")
        print("Total SIMILAR_TO relationships:", cnt[0]["rels"] if cnt else 0)
        return total_created

class KnowledgeGraph(ChunkNodes):
    def __init__(self, kg: Neo4jGraph, EMB_DIM: int = EMB_DIM) -> None:
        super().__init__()
        self.kg = kg
        self.EMB_DIM = EMB_DIM
        self.health_check = EmbeddingHealthCheck(kg)

    # Axuliar graph building functions
    def _setup_constraints(self, kg):
        unique_paper_id = """
            CREATE CONSTRAINT unique_paper IF NOT EXISTS
                FOR (p:Paper) REQUIRE p.paperId IS UNIQUE;
        """

        unique_chunk_id = """
            CREATE CONSTRAINT unique_chunk IF NOT EXISTS
                FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
        """

        unique_task_name = """
            CREATE CONSTRAINT unique_task IF NOT EXISTS
                FOR (t:Task) REQUIRE t.name IS UNIQUE;
        """

        unique_dataset_name = """
            CREATE CONSTRAINT unique_dataset IF NOT EXISTS
            FOR (d:Dataset) REQUIRE d.name is UNIQUE;
        """

        unique_model_name = """
            CREATE CONSTRAINT unique_model IF NOT EXISTS
            FOR (m:Model) REQUIRE m.name is UNIQUE;
        """

        self.kg.query(unique_paper_id)
        print("Unique paper constraint ready")
        self.kg.query(unique_chunk_id)
        print("Unique chunk constraint ready")
        self.kg.query(unique_task_name)
        print("Unique task name constraint ready")
        self.kg.query(unique_dataset_name)
        print("Unique dataset name constraint ready")
        self.kg.query(unique_model_name)
        print("Unique model name constraint ready")

        print ("All constraints ensured.")

    def _setup_indexes(self, kg):
        # create chunk vector index
        kg.query(f"""
            CREATE VECTOR INDEX `{CHUNK_INDEX_NAME}` IF NOT EXISTS
            FOR (c:Chunk) on (c.embedding)
            OPTIONS {{ 
                indexConfig: {{
                `vector.dimensions`: {self.EMB_DIM},
                `vector.similarity_function`: 'cosine'
                }}
            }}
        """)

        print(f"Vector index with name {CHUNK_INDEX_NAME} ready.")

        # create paper vector index
        kg.query(f"""
            CREATE VECTOR INDEX `{PAPER_INDEX_NAME}` IF NOT EXISTS
            FOR (p:Paper) on (p.embedding)
            OPTIONS {{
                indexConfig: {{
                `vector.dimensions`: {self.EMB_DIM},
                `vector.similarity_function`: 'cosine'
                }}
            }}
        """)

        print(f"Vector index with name {PAPER_INDEX_NAME} ready.")

        # create custom text index
        kg.query(f"""
            CREATE FULLTEXT INDEX {TEXT_INDEX} IF NOT EXISTS
            FOR (c:{NODE_FOR_TEXT_INDEX}) ON EACH [c.{PROPERTY_FOR_TEXT_INDEX}]
        """)

        print(f"Text index with name {TEXT_INDEX} ready")

    def rebuild_indexes(self, indexes: list):
        """Rebuild vector indexes"""
        import time
    
        # drop existing indexes
        for index in indexes:
            try:
                self.kg.query(f"DROP INDEX {index} IF EXISTS")
                print(f"Dropped index: {index}")
            except Exception as e:
                print(f"Error dropping index {index}: {e}")

            time.sleep(5)

            # build indexes
            self._setup_indexes(kg)

            # verify state 
            status = kg.query("""
                SHOW INDEXES
                YIELD name, state, populationPercent
                WHERE name IN $index_names
                RETURN name, state, populationPercent
            """, params={"index_names": indexes})

            print("Rebuild complete")
            for s in status:
                print(f"  {s['name']}: {s['state']}, {s['populationPercent']}% populated")

        return status

    # paper level ingestion
    def _paper_nodes(self, paper_meta):
        """Create all Paper, Dataset, Model nodes and relationships"""
        query = """
        MERGE (p:Paper {paperId: $paper_id})
        SET p.title = $paper_node.title,
            p.year = $paper_node.year,
            p.url = $paper_node.url
        WITH p
        UNWIND $tasks AS taskName
            MERGE (t:Task {name: taskName})
            MERGE (p)-[:ADDRESSES_TASK]->(t)

        WITH p
        UNWIND $datasets AS datasetName
            MERGE (d:Dataset {name: datasetName})
            MERGE (p)-[:USES_DATASET]->(d)

        WITH p
        MERGE (m:Model {name: $model})
        MERGE (p)-[:EXPLAINS_MODEL]->(m)

        FOREACH (_ IN CASE WHEN $derived_from IS NULL THEN [] ELSE [1] END |
            MERGE (base:Model {name: $derived_from})
            MERGE (m)-[:DERIVED_FROM]->(base)
        )
        """
        
        self.kg.query(query, params=paper_meta)

    # chunk level ingestion
    def _chunk_nodes(self):
        """Build Chunk nodes and relationships"""
        chunks = self.all_chunks

        query = """
        MERGE (chunk:Chunk {chunkId: $chunk.chunk_id})
            ON CREATE SET
                chunk.text = $chunk.text,
                chunk.embedding = $chunk.embedding,
                chunk.embeddingModel = $embedding_model,
                chunk.embeddingUpdatedAt = datetime()
        WITH chunk
        MATCH (p:Paper {paperId: $chunk.paper_id})
        MERGE (p)-[:HAS_CHUNK]->(chunk)
        RETURN chunk
        """

        created = 0
        for chunk in chunks:
            params = {
                "chunk": chunk,
                "embedding_model": chunk.get("embedding_model") or EMBEDDING_MODEL
            }
            self.kg.query(query, params=params)
            created += 1

        print(f"[chunk_nodes] Created/merged {created} chunks")

    def populate_graph(self):
        self._setup_constraints(self.kg)
        # create indexes
        self._setup_indexes(self.kg)

        # nodes construction
        papers_metadata = self._paper_level_metadata()
        self._chunk_level_metadata()

        for paper_meta in papers_metadata:
            self._paper_nodes(paper_meta)
            total = self.kg.query("""
                MATCH (n)
                RETURN count(n) as nodeCount
                """)
            print(f"Built {total} nodes")

        self._chunk_nodes()

        """if self.calculate_embeddings:
            self.calculate_paper_level_embeddings()
            self.add_paper_embeddings()"""
            
        self._chunks_knn_relationships(self.kg)
        # perform health check
        self.health_check.perform_health_check(self.kg)

    def build_graph(self, 
        paper_paths: list,  
        json_metadata = json.load(open('knowledge_graphs/data.json')),
        calculate_embeddings: bool = True, 
    ):
        self.calculate_embeddings = calculate_embeddings
        self.paper_paths = paper_paths
        self.json_metadata = json_metadata

        return self.populate_graph()

class EmbeddingHealthCheck(KnowledgeGraph):
    def __init__(self, kg) -> None:
        self.kg = kg
        self.k = 7

    def check_missing_or_outdated_nodes(self, label="Chunk"):
        """
        Check for missing embeddings or embeddings with an old model
        Checks on the nodes named as label, "Chunk" by default
        """
        self.label = label

        model = EMBEDDING_MODEL
        print(f"Using embedding model: {model}")

        if label == "Chunk":
            id_prop = "chunkId"
        elif label == "Paper":
            id_prop = "paperId"
        else:
            id_prop = "id"
        
        query = """
        MATCH (n:%s)
        WHERE 
            n.embedding is NULL 
            OR n.embedding = []
            OR size(n.embedding) = 0
            OR n.embeddingModel <> $model_name
        RETURN n.%s AS id, n.text AS text
        """ % (label, id_prop)

        self.records = self.kg.query(query, params={"model_name": model})

        print(f"Damaged records: {len(self.records)}")

        return self.records

    def calculate_damaged_embeddings(self, records):
        """Re calculate embeddings for missing or outdated nodes"""
        self.updated_ids = []

        if records:
            self.records = records
        
        if self.label == "Paper":
            self.calculate_paper_level_embeddings()
            return self.add_paper_embeddings()

        count = 0
        for rec in self.records:
            embedding = EMBEDDER.embed_documents([rec["text"]])
            emb = embedding[0]

            if self.label == "Chunk":
                id_prop = "chunkId"
            elif self.label == "Paper":
                id_prop = "paperId"
            else:
                id_prop = "id"
            
            self.kg.query("""
                MATCH (n:{label} {{{id_prop}: $id}})
                SET n.embedding = $embedding,
                    n.embeddingModel = $model_name,
                    n.embeddingUpdatedAt = datetime()
            """.format(label=self.label, id_prop=id_prop),
            params={
                "id": rec["id"],
                "embedding": emb,
                "model_name": EMBEDDING_MODEL
            })

            self.updated_ids.append(rec["id"])
            count += 1

        print(f"Fixed {count} node embeddings")

    def rebuild_modified_knn_relationships(self, k: int = None):
        """Rebuild ONLY MODIFIED KNN similarity relationships"""
        if not hasattr(self, "updated_ids") or not self.updated_ids:
            print("No updated embeddings... Skipping KNN relationships.")
            return

        print(f"Recomputing KNN for {len(self.updated_ids)} updated nodes")

        if k:
            self.k = k

        kg = self.kg
        # remove SIMILAR_TO edges for modified nodes
        kg.query(
            """
            MATCH (c:Chunk)-[r:SIMILAR_TO]->(:Chunk)
            WHERE c.chunkId IN $ids
            DELETE r
            """,
            params={"ids": self.updated_ids}
        )

        index_name = CHUNK_INDEX_NAME
        
        # ? Eventually this should be batch size 
        # Rebuild edges only for those nodes
        query = f"""
        UNWIND $ids as chunk_id
        MATCH (c:Chunk {{chunkId: chunk_id}})
        CALL db.index.vector.queryNodes(
            '{index_name}',
            $k,
            c.embedding
        ) YIELD node, score
        WHERE node <> c
        MERGE (c)-[r:SIMILAR_TO]->(node)
        SET r.score = score
        """

        kg.query(query, params={"ids": self.updated_ids, "k": self.k})

        print(f"KNN relationship rebuilt for {len(self.updated_ids)} nodes")

    def rebuild_all_knn_relationships(self, k: int = None):
        """Rebuild ALL KNN similarity relationships when needed"""
        kg = self.kg
       
        if k:
            self.k = k

        # delete all SIMILAR_TO relationships
        kg.query("""
        MATCH (:Chunk)-[r:SIMILAR_TO]-(:Chunk)
        DELETE r
        """)

        self._chunks_knn_relationships(kg)

    def perform_health_check(self, kg, label: str = None):
        """perform embedding health check and re calculate if necessary"""
        if label:
            self.label = label
            records = self.check_missing_or_outdated_nodes(label)
        else:
            records = self.check_missing_or_outdated_nodes()
        

        if records:
            print("Re generating damaged embeddings...")
            self.calculate_damaged_embeddings(records)
            self.rebuild_modified_knn_relationships()

            return records # for logs

        print("All embeddings healthy.")     
        return [] 

def populate_with_uploads(metadata, text):
    metadata["tasks"] = []
    metadata["datasets"] = []
    metadata["model"] = metadata.get("paper", "Unknown Model")
    metadata["derived_from"] = None
    
    paper_paths = [{metadata["model"].lower(): metadata["pdf_path"]}]
    json_metadata = [metadata]

    graph = KnowledgeGraph(kg)
    graph.build_graph(
        paper_paths=paper_paths, 
        json_metadata=json_metadata, 
        calculate_embeddings=False
    )

# test functions
def run_graph_population(rebuild_all: bool = False):
    graph = KnowledgeGraph(kg)
    if rebuild_all:
        indexes = [CHUNK_INDEX_NAME, PAPER_INDEX_NAME, TEXT_INDEX]
        graph.rebuild_indexes(indexes=indexes)
    graph.build_graph(paper_paths=paper_paths, calculate_embeddings=False)

def test_rag_queries():
    rag = RAGQuery(kg)
    query = "How does GPT works?"
    rsp, metadata = rag.retrieve(query, top_k=2, expand=False, hops=1) 

    return rsp, metadata

async def llm_query():
    from rag import GraphRetrieval
    rag = GraphRetrieval(kg)
    query = "How does GPT works?"
    rsp = await rag.generate_response(query, top_k=TOP_K, expand=True, hops=1)

    return rsp

def test_extract_text_from_pdf(pdf_path: str):
    g = KnowledgeGraph(kg)
    text = g._extract_text_from_pdf(pdf_path)
    return text

'''if __name__ == "__main__":
    #run_graph_population(rebuild_all=True)

    #rsp, metadata = test_rag_queries()
    #print(rsp, "\n", metadata)

    rsp = asyncio.run(llm_query())
    print(rsp)'''