import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()


def load_documents(data_dir: str):
    documents = []

    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith(".pdf"):
                filepath = os.path.join(root, filename)
                print(f"Loading: {filepath}")

                loader = PyPDFLoader(filepath)
                documents.extend(loader.load())
    
    return documents
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks
def embed_and_store(chunks, persist_dir: str):
    print("Initializing OpenAI embeddings...")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("Embedding chunks and storing in ChromaDB...")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="designmate_docs"
    )
    
    print(f"Done. Vectorstore saved to: {persist_dir}")
    return vectorstore

if __name__ == "__main__":
    DATA_DIR = "data"
    PERSIST_DIR = "vectorstore"

    documents = load_documents(DATA_DIR)

    if not documents:
        print("No PDFs found. Please add PDFs to data/ folder.")
    else:
        chunks = chunk_documents(documents)
        embed_and_store(chunks, PERSIST_DIR)
        print("Ingestion complete!")