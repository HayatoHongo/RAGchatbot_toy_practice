from langchain_community.embeddings import OpenAIEmbeddings
from chromadb import Client
from chromadb.config import Settings
from pathlib import Path

# プロジェクトルート直下の chroma_db
ROOT    = Path(__file__).parent.parent
DB_PATH = str(ROOT / "chroma_db")

def build_vector_store(chunks, metadatas):
    emb = OpenAIEmbeddings(model="text-embedding-ada-002")

    # legacyキーを渡さず persist_directory のみ
    client = Client(Settings(persist_directory=DB_PATH))
    col    = client.get_or_create_collection("internal_docs")

    embeddings = emb.embed_documents(chunks)
    col.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    return emb
