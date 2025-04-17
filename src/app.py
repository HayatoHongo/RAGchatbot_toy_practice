import os, sqlite3
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# ← PersistentClient は使わず、新 Client(Settings) だけにします
from chromadb import Client
from chromadb.config import Settings

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

from data_loader import docs
from preprocess import chunk_texts

# 環境変数
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# プロジェクトルート＆DBパス
ROOT    = Path(__file__).parent.parent
DB_PATH = str(ROOT / "chroma_db")

# ドキュメントのチャンク化 & 埋め込み準備
chunks, metadatas = chunk_texts(docs)
emb = OpenAIEmbeddings(model="text-embedding-ada-002")

# 新 API：legacyキーを渡さず persist_directory だけ指定
client = Client(Settings(persist_directory=DB_PATH))
vectorstore = Chroma(
    client=client,
    collection_name="internal_docs",
    embedding_function=emb
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm       = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain  = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Streamlit UI
st.set_page_config(page_title="社内文書チャットボット", layout="wide")
st.title("📄 社内文書チャットボット")
st.write("社内規程やガイドラインに関する質問にお答えします。")

query = st.text_input("質問を入力してください:", "")
if st.button("送信") and query:
    res = qa_chain({"query": query})

    st.subheader("💬 回答")
    st.write(res["result"])

    st.subheader("📚 参照元ドキュメント")
    if res.get("source_documents"):
        for doc in res["source_documents"]:
            sid     = doc.metadata.get("source", "unknown")
            snippet = doc.page_content.strip().replace("\n", " ")[:100]
            st.markdown(f"- **{sid}**: {snippet}...")
    else:
        st.write("参照元が見つかりませんでした。")
