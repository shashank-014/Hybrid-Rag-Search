from datetime import datetime
from pathlib import Path

import streamlit as st

from config import DATA_DIR, DELETED_DIR, ensure_dirs
from indexing.chunking import chunk_documents
from indexing.vector_store import index_documents, load_faiss_index
from ingestion.loaders import load_sources
from rag.answer_generator import stream_answer
from rag.context_builder import build_context
from rag.memory import create_memory, load_memory_text, save_turn
from rag.summarizer import summarize_documents
from retrieval.query_rewriter import rewrite_query
from retrieval.query_router import route_query
from retrieval.reranker import rerank_documents
from retrieval.semantic_search import search_documents
from web.tavily_search import search_web

ROUTE_LABELS = {
    "document": "📄 Document answer",
    "web": "🌐 Web answer",
    "hybrid": "🔀 Hybrid answer",
}



def _archive_existing(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y-%m-%d")
    archived = DELETED_DIR / f"data_documents_{stamp}_prev_{path.name}"
    path.replace(archived)



def _save_uploaded_files(uploaded_files: list) -> list[Path]:
    ensure_dirs()
    saved_paths: list[Path] = []

    for uploaded in uploaded_files:
        target = DATA_DIR / uploaded.name
        _archive_existing(target)
        target.write_bytes(uploaded.getbuffer())
        saved_paths.append(target)

    return saved_paths



def _get_memory():
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = create_memory()
    return st.session_state.conversation_memory



def _reset_memory() -> None:
    st.session_state.conversation_memory = create_memory()
    st.session_state.chat_messages = []



def _get_indexed_titles() -> list[str]:
    return sorted([path.name for path in DATA_DIR.glob("*") if path.is_file()])



def _index_sources(file_paths: list[Path], wiki_topics: list[str]) -> int:
    records = load_sources(file_paths, wiki_topics)
    if not records:
        return 0

    st.session_state.ingested_records = records
    chunks = chunk_documents(records)
    store = index_documents(chunks)
    st.session_state.vector_store = store
    return len(chunks)



def _load_store_from_disk():
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = load_faiss_index()
    return st.session_state.vector_store



def _run_query(query: str, use_web: bool) -> dict[str, object]:
    store = _load_store_from_disk()
    memory = _get_memory()
    route = route_query(query)
    rewritten = rewrite_query(query)

    doc_hits = []
    if store and route in {"document", "hybrid"}:
        doc_hits = search_documents(rewritten["vector_query"], store)
        doc_hits = rerank_documents(rewritten["vector_query"], doc_hits, top_k=5)

    web_hits = []
    if use_web and route in {"web", "hybrid"}:
        web_hits = search_web(rewritten["web_query"], top_k=5)

    payload = build_context(doc_hits, web_hits, route)
    summaries = summarize_documents(doc_hits, max_items=3)

    return {
        "route": route,
        "route_label": ROUTE_LABELS.get(route, route),
        "rewritten": rewritten,
        "doc_evidence": payload["doc_evidence"],
        "web_evidence": payload["web_evidence"],
        "context": payload["context"],
        "summaries": summaries,
        "memory_text": load_memory_text(memory),
    }



def _render_top_sources(summaries: list[dict[str, object]]) -> None:
    st.subheader("Top Sources")
    if not summaries:
        st.caption("No document summaries available for this answer.")
        return

    for item in summaries:
        score = item.get("similarity_score")
        score_text = f" | FAISS score: {score:.4f}" if isinstance(score, float) else ""
        st.markdown(f"**{item['title']}**{score_text}")
        st.write(item["summary"])



def _render_doc_evidence(doc_evidence: list[dict[str, object]], summaries: list[dict[str, object]]) -> None:
    _render_top_sources(summaries)
    st.divider()

    if not doc_evidence:
        st.caption("No document evidence used.")
        return

    for item in doc_evidence:
        meta = item["metadata"]
        score = meta.get("similarity_score")
        score_text = f"{score:.4f}" if isinstance(score, float) else "n/a"
        st.markdown(f"**{item['citation']}**")
        st.caption(
            f"Title: {meta.get('document_title', 'Unknown')} | "
            f"Chunk: {meta.get('chunk_index', 'n/a')} | "
            f"FAISS score: {score_text}"
        )
        if "rerank_score" in meta:
            st.caption(f"Rerank score: {meta['rerank_score']:.4f}")
        st.write(item["content"])



def _render_web_evidence(web_evidence: list[dict[str, object]]) -> None:
    if not web_evidence:
        st.caption("No web evidence used.")
        return

    for item in web_evidence:
        st.markdown(f"**{item['citation']}**")
        st.write(item["snippet"])
        if item["url"]:
            st.markdown(item["url"])



def run_app() -> None:
    ensure_dirs()
    st.set_page_config(page_title="Hybrid RAG Search", layout="wide")
    st.title("Hybrid Multi-Document RAG Search")
    st.caption("Internal documents plus real-time web search in one chat flow.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    with st.sidebar:
        st.header("Sources")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md", "rst"],
            accept_multiple_files=True,
        )
        wiki_input = st.text_area("Wikipedia topics", placeholder="One topic per line")
        use_web = st.toggle("Enable Tavily search", value=True)

        if st.button("Index sources", use_container_width=True):
            file_paths = _save_uploaded_files(uploaded_files or [])
            wiki_topics = [line.strip() for line in wiki_input.splitlines() if line.strip()]
            chunk_count = _index_sources(file_paths, wiki_topics)
            if chunk_count:
                st.success(f"Indexed {chunk_count} chunks.")
            else:
                st.warning("No sources were indexed.")

        if st.button("Reset chat", use_container_width=True):
            _reset_memory()
            st.success("Conversation memory cleared for this session.")

        st.subheader("Indexed files")
        indexed_titles = _get_indexed_titles()
        if indexed_titles:
            for title in indexed_titles:
                st.write(f"- {title}")
        else:
            st.caption("No local files indexed yet.")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask about your documents, the web, or both")
    if not query:
        st.info("Upload files or add Wikipedia topics, then ask a question.")
        return

    st.session_state.chat_messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    result = _run_query(query, use_web)

    with st.chat_message("assistant"):
        st.markdown(result["route_label"])
        if result["rewritten"]["was_rewritten"]:
            st.caption(f"Rewritten query: {result['rewritten']['rewritten_query']}")

        answer_tab, doc_tab, web_tab = st.tabs(["Answer", "Document Evidence", "Web Evidence"])
        with answer_tab:
            answer = st.write_stream(
                stream_answer(query, result["context"], result["memory_text"])
            )

        with doc_tab:
            _render_doc_evidence(result["doc_evidence"], result["summaries"])

        with web_tab:
            _render_web_evidence(result["web_evidence"])

    save_turn(_get_memory(), query, answer)
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
