# src/app.py

import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from chromadb import PersistentClient

# --- 環境変数読み込み ---
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- モジュールインポート ---
from data_loader import docs
from preprocess import chunk_texts
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from qa_chain import QA_PROMPT  # カスタムプロンプトテンプレート

# --- プロジェクトルート＆DBパス ---
ROOT    = Path(__file__).parent.parent
DB_PATH = str(ROOT / "chroma_db")

# --- RAG 構築 ---
# ドキュメントのチャンク化
chunks, metadatas = chunk_texts(docs)
# 埋め込みモデル
emb = OpenAIEmbeddings(model="text-embedding-ada-002")
# PersistentClient を共有
client      = PersistentClient(path=DB_PATH)
# Chroma retriever
vectorstore = Chroma(
    client=client,
    collection_name="internal_docs",
    embedding_function=emb
)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
# QA チェーン
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

# --- Streamlit UI設定 ---
st.set_page_config(page_title="社内文書チャットボット", layout="wide")
st.title("📄 社内文書チャットボット")
st.write("社内規程やガイドラインに関する質問にお答えします。")

# --- デバッグ用サイドバー ---
st.sidebar.header("🔧 デバッグモード")
show_db    = st.sidebar.checkbox("🔍 DBパス確認")
show_ch    = st.sidebar.checkbox("🔍 Chroma 直読")
show_cnt   = st.sidebar.checkbox("登録チャンク数を表示")
show_hits  = st.sidebar.checkbox("Retriever ヒットを表示")
show_raw   = st.sidebar.checkbox("生レスポンスを表示")
show_prompt= st.sidebar.checkbox("送信プロンプトを表示")

# --- DBパス確認 ---
if show_db:
    st.sidebar.write("🏠 カレントディレクトリ:", os.getcwd())
    st.sidebar.write("📁 DB_PATH:", DB_PATH)
    st.sidebar.write("📂 DBフォルダ存在？", os.path.exists(DB_PATH))
    if os.path.exists(DB_PATH):
        st.sidebar.write("📂 chroma_db 配下:", os.listdir(DB_PATH))
        try:
            dbfile = Path(DB_PATH) / "chroma.sqlite3"
            conn   = sqlite3.connect(str(dbfile))
            cur    = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [t[0] for t in cur.fetchall()]
            st.sidebar.write("📋 テーブル一覧:", tables)
            if "internal_docs" in tables:
                cur.execute("SELECT COUNT(*) FROM internal_docs")
                st.sidebar.write("🔢 internal_docs レコード数:", cur.fetchone()[0])
            conn.close()
        except Exception as e:
            st.sidebar.write("❌ SQLite 読み取り失敗:", e)

# --- Chroma 直読 ---
if show_ch:
    try:
        cols = [c.name for c in client.list_collections()]
        st.sidebar.write("🔍 コレクション一覧:", cols)
        if "internal_docs" in cols:
            col = client.get_collection(name="internal_docs")
            st.sidebar.write("🔢 internal_docs レコード数:", col.count())
    except Exception as e:
        st.sidebar.write("❌ Chroma 直読失敗:", e)

# --- 質問入力＆プロンプト表示 ---
query = st.text_input("質問を入力してください:", "")
if show_prompt and query:
    docs_for_prompt = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs_for_prompt])
    prompt = QA_PROMPT.format(context=context, question=query)
    st.sidebar.subheader("📨 送信プロンプト")
    st.sidebar.text_area("", prompt, height=300)

# --- 質問の実行 ---
if st.button("送信") and query:
    with st.spinner("回答を生成中…"):
        res = qa_chain({"query": query})

    # 回答表示
    st.subheader("💬 回答")
    st.write(res["result"])

    # 参照元表示
    st.subheader("📚 参照元ドキュメント")
    if res.get("source_documents"):
        for doc in res["source_documents"]:
            sid     = doc.metadata.get("source", "unknown")
            snippet = doc.page_content.strip().replace("\n", " ")[:100]
            st.markdown(f"- **{sid}**: {snippet}...")
    else:
        st.write("参照元が見つかりませんでした。")

    # --- デバッグ情報 ---
    if show_cnt:
        try:
            cnt = vectorstore._collection.count()
        except:
            cnt = "取得失敗"
        st.sidebar.write(f"登録チャンク数：{cnt}")

    if show_hits:
        st.sidebar.subheader("Retriever.get_relevant_documents() 結果")
        hits = retriever.get_relevant_documents(query)
        if hits:
            for i, d in enumerate(hits, 1):
                sid  = d.metadata.get("source", "unknown")
                text = d.page_content.strip().replace("\n", " ")[:80]
                st.sidebar.write(f"{i}. [{sid}] {text}...")
        else:
            st.sidebar.write("ヒットなし")

    if show_raw:
        st.sidebar.subheader("生レスポンス JSON")
        st.sidebar.json(res)
