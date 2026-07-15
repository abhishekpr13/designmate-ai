# DesignMate AI

A question-answering assistant for mechanical and CAD design engineering. Ask a question like
*"What is the minimum bend radius for sheet metal?"* and it finds the answer from trusted
reference guides (sheet metal, GD&T, CAD drafting best practices) instead of guessing.

It uses **RAG** (Retrieval-Augmented Generation): the reference PDFs are turned into searchable
vectors, and each question pulls the most relevant chunks of text.

## How it works

1. **Ingestion** (`src/ingestion/ingest.py`) — reads the PDFs in `data/`, splits them into small
   chunks, turns each chunk into an embedding with OpenAI, and stores them in a local ChromaDB
   vector store.
2. **Retrieval** (`src/retrieval/retrieve.py`) — takes a question and returns the most similar
   chunks from the vector store.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then add your real OPENAI_API_KEY
```

## Usage

```bash
# 1. Build the vector store from the PDFs in data/
python src/ingestion/ingest.py

# 2. Search it
python src/retrieval/retrieve.py
```

## Project structure

```
data/          Reference PDFs (sheet metal, GD&T, CAD drafting)
src/ingestion  PDF loading, chunking, embedding, storing
src/retrieval  Similarity search over the vector store
src/agents     (planned) query intent classifier
vectorstore/   Generated ChromaDB store (not committed)
```

## Status

Working: PDF ingestion and chunk retrieval.
Planned: answer generation with citations, intent classifier, API, and a web UI.
