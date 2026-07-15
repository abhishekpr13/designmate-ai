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
        collection_name="designmate_docs",
        # Must match the metric used when the store was built in ingest.py.
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Vectorstore loaded from: {persist_dir}")
    return vectorstore
def retrieve_chunks(vectorstore, query: str, k: int = 4, min_score: float = 0.2):

    print(f"Searching for: {query}")

    # similarity_search_with_relevance_scores returns a list of (Document, score)
    # tuples. The score is normalized between 0 and 1 where 1 = most similar.
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)

    # Drop weak matches. This is the guardrail that stops us from answering
    # from irrelevant text when the docs don't actually cover the question.
    strong_results = [(doc, score) for doc, score in results if score >= min_score]

    print(f"Found {len(results)} chunks, {len(strong_results)} above threshold ({min_score})")

    return strong_results

if __name__ == "__main__":
    PERSIST_DIR = "vectorstore"
    
    # Load existing vectorstore
    vectorstore = load_vectorstore(PERSIST_DIR)
    
    # Test search
    query = "What is the minimum bend radius for sheet metal?"
    results = retrieve_chunks(vectorstore, query)
    
    # Print results. Each item is now a (document, score) tuple.
    for i, (doc, score) in enumerate(results):
        similarity_pct = score * 100
        print(f"\n--- Chunk {i+1} ---")
        print(f"Similarity: {similarity_pct:.1f}%")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Page: {doc.metadata.get('page', 'unknown')}")
        print(f"Content: {doc.page_content[:200]}...")