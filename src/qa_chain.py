# src/qa_chain.py

from pathlib import Path
from chromadb import PersistentClient
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

# プロジェクトルート＆DBパス
ROOT    = Path(__file__).parent.parent
DB_PATH = str(ROOT / "chroma_db")

def get_qa_chain(emb=None):
    """
    build_vector_store で作った PersistentClient を流用し、
    Retriever + RetrievalQA を返す。
    """
    if emb is None:
        emb = OpenAIEmbeddings(model="text-embedding-ada-002")

    client      = PersistentClient(path=DB_PATH)
    vectorstore = Chroma(
        client=client,
        collection_name="internal_docs",
        embedding_function=emb
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
