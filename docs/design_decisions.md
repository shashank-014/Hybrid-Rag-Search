# Design Decisions

## Why a modular layout

The project is split by pipeline stage so ingestion, indexing, retrieval, web search, RAG assembly, and UI can evolve independently without turning the app into one large script.

## Why FAISS

FAISS is a good fit for local semantic retrieval because it is fast, widely used, and simple to persist for a Streamlit-based workflow.

## Why query routing

Not every question needs live web search. A router keeps costs down, avoids noisy external context, and makes it easier to explain why the system picked a specific retrieval path.

## Why keep web search separate

Tavily access sits behind its own module so external retrieval can be tested, swapped, or rate-limited without tangling the internal document pipeline.

## Why keep reranking optional

Cross-encoder reranking improves relevance but adds latency. Keeping it as a dedicated module makes it easy to add later when quality needs justify the cost.

## Why context building is its own layer

Hybrid RAG quality usually depends on how evidence is balanced, deduplicated, and ordered before answer generation. Pulling this into a separate module keeps that logic explicit.

## Why Streamlit for Phase 1

Streamlit is the fastest way to ship a usable internal search UI while keeping the Python-first stack simple.

## Security note

Any future secrets must be read with `st.secrets["KEY_NAME"]`. No keys should be hardcoded in the repo.
