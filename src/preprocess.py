# src/preprocess.py

from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_texts(docs):
    splitter  = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks    = []
    metadatas = []
    for doc in docs:
        for chunk in splitter.split_text(doc["text"]):
            chunks.append(chunk)
            metadatas.append({"source": doc["id"]})
    return chunks, metadatas
