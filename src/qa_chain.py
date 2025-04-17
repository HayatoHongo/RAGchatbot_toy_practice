# src/qa_chain.py

from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pathlib import Path
from chromadb import PersistentClient

# プロジェクトルートと DB パス
ROOT    = Path(__file__).parent.parent
DB_PATH = str(ROOT / "chroma_db")

# カスタムプロンプトテンプレート
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "以下の参照ドキュメントを参考に、質問に回答してください。必ず、参照ドキュメントの内容に言及する形で、回答を生成してください。\n"
        "---\n"
        "{context}\n"
        "---\n"
        "質問: {question}\n"
        "回答:"
    )
)

def get_qa_chain(emb):
    """
    埋め込みモデル emb と PersistentClient を使い、
    質問に対して参照ドキュメントを考慮した回答を生成する RetrievalQA を返す。
    """
    # PersistentClient で同一 DB を利用
    client = PersistentClient(path=DB_PATH)

    # Chroma retriever を生成
    vectorstore = Chroma(
        client=client,
        collection_name="internal_docs",
        embedding_function=emb
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # LLM と RetrievalQA チェーン (カスタムプロンプトを適用)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    return qa_chain
