import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = os.path.join(os.path.dirname(__file__), "../docs")
CHROMA_PATH = "./chroma_db"

def initialize_vector_db():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="zepto_policies")
    
    if collection.count() == 0:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        doc_files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt")))
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, filepath in enumerate(doc_files):
            filename = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                documents.append(content)
                metadatas.append({"source": filename})
                ids.append(f"doc_{idx+1}")
                
        embeddings = model.encode(documents).tolist()
        
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    return collection

def retrieve_top_chunks(query: str, k=3):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name="zepto_policies")
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )
    return results