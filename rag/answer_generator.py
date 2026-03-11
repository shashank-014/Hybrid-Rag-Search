import os
from collections.abc import Iterator

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]



def _build_messages(query: str, context: str, memory_text: str = "") -> list:
    system_prompt = (
        "You are a grounded RAG assistant. Use only the supplied context. "
        "If the answer is missing, say so plainly. Cite every factual claim using the provided labels."
    )
    user_prompt = (
        f"Conversation memory:\n{memory_text or 'None'}\n\n"
        f"Question:\n{query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Write a concise answer using only this context. "
        "Use citations in this style: [Doc] filename - chunk3 or [Web] Tavily - article title."
    )
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]



def stream_answer(query: str, context: str, memory_text: str = "") -> Iterator[str]:
    if "GROQ_API_KEY" not in st.secrets:
        yield "GROQ_API_KEY is missing in Streamlit secrets."
        return

    if not context.strip():
        yield "I could not find enough evidence to answer that question yet."
        return

    llm = ChatGroq(model="llama3-70b-8192", temperature=0.2)
    for chunk in llm.stream(_build_messages(query, context, memory_text)):
        if chunk.content:
            yield chunk.content



def generate_answer(query: str, context: str, memory_text: str = "") -> str:
    return "".join(stream_answer(query, context, memory_text))
