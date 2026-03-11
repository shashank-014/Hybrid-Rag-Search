# Hybrid Multi-Document RAG Search Engine with Real-Time Web Search

## Overview

This project combines internal document retrieval with real-time web search through a hybrid Retrieval-Augmented Generation pipeline. It lets users ask questions against uploaded files, Wikipedia content, and live web results in one Streamlit interface while keeping evidence visible through citations, summaries, and retrieval scores.

## Key Features

- Multi-document ingestion for PDF, TXT, Markdown, and Wikipedia sources
- FAISS semantic search over indexed internal content
- Query routing for document, web, and hybrid question paths
- Tavily real-time web search for fresh external context
- RAG answer generation with document and web citations
- Streamlit chatbot interface with evidence tabs

## Advanced Capabilities

- Conversation memory for follow-up questions in the same session
- Cross-encoder reranking for stronger retrieval quality
- Query rewriting for short or vague prompts
- Document summarization for top retrieved sources
- Retrieval score transparency in the evidence view

## System Architecture

The main pipeline flows like this:

document ingestion -> chunking -> vector indexing -> semantic retrieval -> web search -> context assembly -> answer generation

Internal files are cleaned, chunked, embedded, and stored in a FAISS index. User queries are routed between internal retrieval, live web retrieval, or a balanced hybrid path. The system then assembles the best evidence into a grounded prompt and generates a cited answer in the Streamlit UI.

## Project Structure

```text
hybrid-rag-search/
|-- app.py
|-- config.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- ingestion/
|   |-- schema.py
|   |-- loaders.py
|   |-- cleaner.py
|-- indexing/
|   |-- chunking.py
|   |-- vector_store.py
|-- retrieval/
|   |-- semantic_search.py
|   |-- reranker.py
|   |-- query_router.py
|   |-- query_rewriter.py
|-- web/
|   |-- tavily_search.py
|-- rag/
|   |-- context_builder.py
|   |-- answer_generator.py
|   |-- citation_formatter.py
|   |-- memory.py
|   |-- summarizer.py
|-- ui/
|   |-- streamlit_ui.py
|-- evaluation/
|   |-- test_queries.py
|   |-- evaluation_report.md
|-- docs/
|   |-- architecture.md
|   |-- design_decisions.md
|-- data/
|   |-- documents/
|       |-- sample_document.txt
|-- faiss_index/
|   |-- index_placeholder.txt
```

## Installation

1. Create and activate a Python 3 environment.
2. Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit app from the repository root:

```bash
streamlit run app.py
```

## Example Queries

Document queries:
- What do the uploaded transformer notes say about attention?
- Summarize the key points from the indexed report.

Web queries:
- What are the latest developments in retrieval augmented generation?
- What is the latest news about FAISS?

Hybrid queries:
- Compare the uploaded notes with recent web updates on retrieval augmented generation.
- Use the indexed report and recent web results to explain current RAG trends.

## Deployment

This repository is prepared for Streamlit Community Cloud deployment.

1. Push the repository to GitHub.
2. Create a new Streamlit Community Cloud app pointing to this repository.
3. Set `app.py` as the entrypoint.
4. Add the required secrets in the Streamlit app settings.

The visible placeholder files in `data/documents/` and `faiss_index/` keep those folders present on GitHub before real data or index files are generated.

## Environment Variables

The app requires these keys:

- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

Store them in Streamlit secrets, not in source files. The app reads them via `st.secrets["OPENAI_API_KEY"]` and `st.secrets["TAVILY_API_KEY"]`.

## Future Improvements

- Add hybrid BM25 plus vector retrieval for stronger recall
- Add knowledge graph integration for entity-aware search
- Add automated retrieval and answer quality evaluation
- Add richer citation highlighting inside answer text
- Add persistent multi-session chat history
