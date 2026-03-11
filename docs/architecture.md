# Architecture

## Goal

Build a hybrid RAG search system that answers user questions using both internal documents and live web results, while showing where each answer came from and how the answer was assembled.

## Hybrid RAG pipeline

1. Source ingestion loads PDFs, text files, and Wikipedia pages into a shared document schema.
2. Cleaning removes spacing noise, page artifacts, and formatting junk before indexing.
3. Recursive chunking splits documents into overlapping chunks for retrieval.
4. FAISS stores embeddings built with `sentence-transformers/all-MiniLM-L6-v2`.
5. Query routing classifies the request as `document`, `web`, or `hybrid`.
6. Query rewriting expands short or vague requests before retrieval runs.
7. Semantic search pulls the top internal chunks and keeps the FAISS scores in metadata.
8. Cross-encoder reranking improves the order of retrieved chunks before context assembly.
9. Tavily web search adds live external evidence when the route requires it.
10. Context assembly balances internal and web evidence before answer generation.
11. ChatOpenAI generates the final grounded response and Streamlit streams it into the UI.

## Document retrieval flow

- `ingestion/loaders.py` normalizes loader output into the shared fields: `source_id`, `source_type`, `title`, `content`, and `metadata`.
- `indexing/chunking.py` creates chunk-level metadata that includes document title, source type, and chunk index.
- `indexing/vector_store.py` persists the FAISS index locally in `faiss_index/`.
- `retrieval/semantic_search.py` runs FAISS similarity search and keeps the raw scores visible.
- `retrieval/reranker.py` applies `cross-encoder/ms-marco-MiniLM-L-6-v2` so the best chunks move to the top before the answer step.

## Web search integration

- `web/tavily_search.py` calls LangChain's Tavily search tool with `st.secrets["TAVILY_API_KEY"]`.
- Web results are normalized into temporary evidence records with `title`, `snippet`, and `url`.
- The router can send a request directly to web retrieval or combine web and internal retrieval in hybrid mode.

## Context assembly

- `rag/context_builder.py` labels every evidence block with a citation-ready source string.
- Hybrid mode uses a balanced mix of top internal and web evidence.
- The current balancing rule keeps up to 3 document chunks and 2 web results for hybrid questions.
- Total context size is capped so the answer prompt stays manageable.

## Memory and answer flow

- `rag/memory.py` uses `ConversationBufferMemory` to carry follow-up context across turns in the same Streamlit session.
- A new browser session gets a fresh in-memory conversation buffer.
- `rag/answer_generator.py` injects conversation history plus retrieved context into the prompt and streams tokens back to the UI.

## UI transparency

- `ui/streamlit_ui.py` shows the chosen query type with a visual route indicator.
- The Document Evidence tab exposes document title, chunk index, FAISS score, and rerank score.
- Top source summaries give a quick view of the strongest internal evidence before full chunk inspection.
