import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def load_vectorstore(persist_dir: str):
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="designmate_docs"
    )
    
    print(f"Vectorstore loaded from: {persist_dir}")
    return vectorstore
def retrieve_chunks(vectorstore, query: str, k: int = 4):
    
    print(f"Searching for: {query}")
    results = vectorstore.similarity_search(query, k=k)
    
    print(f"Found {len(results)} relevant chunks")
    
    return results

if __name__ == "__main__":
    PERSIST_DIR = "vectorstore"
    
    # Load existing vectorstore
    vectorstore = load_vectorstore(PERSIST_DIR)
    
    # Test search
    query = "What is the minimum bend radius for sheet metal?"
    results = retrieve_chunks(vectorstore, query)
    
    # Print results
    for i, doc in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Page: {doc.metadata.get('page', 'unknown')}")
        print(f"Content: {doc.page_content[:200]}...")