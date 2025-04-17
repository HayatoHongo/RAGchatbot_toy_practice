# src/vector_store.py

from langchain_community.embeddings import OpenAIEmbeddings
from chromadb import PersistentClient
from pathlib import Path

ROOT    = Path(__file__).parent.parent
DB_PATH = str(ROOT / "chroma_db")

def build_vector_store(chunks, metadatas):
    emb = OpenAIEmbeddings(model="text-embedding-ada-002")
    client = PersistentClient(path=DB_PATH)
    existing = [c.name for c in client.list_collections()]
    if "internal_docs" in existing:
        col = client.get_collection(name="internal_docs")
    else:
        col = client.create_collection(name="internal_docs")
    embeddings = emb.embed_documents(chunks)
    col.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return emb
