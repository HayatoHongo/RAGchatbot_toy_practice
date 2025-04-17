import os
from pathlib import Path
from dotenv import load_dotenv
from src.data_loader import docs
from src.preprocess import chunk_texts
from langchain_community.embeddings import OpenAIEmbeddings
from chromadb import Client
from chromadb.config import Settings

# 環境変数
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def main():
    chunks, metadatas = chunk_texts(docs)
    emb = OpenAIEmbeddings(model="text-embedding-ada-002")

    ROOT    = Path(__file__)
    DB_PATH = str(ROOT / "chroma_db")

    client = Client(Settings(persist_directory=DB_PATH))
    col    = client.get_or_create_collection("internal_docs")

    print("👉 登録チャンク数:", col.count())

    query    = "就業規則とは？"
    query_vec= emb.embed_documents([query])[0]
    res = col.query(
        query_embeddings=[query_vec],
        n_results=3,
        include=["documents","metadatas","distances"],
    )
    print(res)
    for i,(d,m,dist) in enumerate(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), 1):
        print(f"\nResult {i}: source={m['source']} (dist={dist})")
        print(d[:100], "…")

if __name__ == "__main__":
    main()
