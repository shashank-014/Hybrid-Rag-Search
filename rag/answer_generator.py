import os
import streamlit as st

from groq import BadRequestError
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

api_key = st.secrets.get("GROQ_API_KEY")

if api_key:
    os.environ["GROQ_API_KEY"] = api_key

MAX_CONTEXT_CHARS = 3500
MAX_MEMORY_CHARS = 1200



def _trim_text(text, limit):
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."



def _build_messages(query, context, memory_text):
    safe_context = _trim_text(context, MAX_CONTEXT_CHARS)
    safe_memory = _trim_text(memory_text, MAX_MEMORY_CHARS)

    system_prompt = f"""
You are a helpful AI assistant.

Use the provided context to answer the user's question.

Context:
{safe_context}

Conversation History:
{safe_memory}

If the answer is not present in the context, say you do not know.
"""
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]



def _invoke_groq(messages):
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.2
    )
    return llm.invoke(messages)



def stream_answer(query, context, memory_text):
    if not api_key:
        yield "GROQ_API_KEY is missing in Streamlit secrets."
        return

    if not context.strip():
        yield "I could not find enough evidence to answer that question yet."
        return

    messages = _build_messages(query, context, memory_text)

    try:
        response = _invoke_groq(messages)
    except BadRequestError:
        fallback_messages = _build_messages(
            query,
            _trim_text(context, 1800),
            _trim_text(memory_text, 400),
        )
        try:
            response = _invoke_groq(fallback_messages)
        except BadRequestError:
            yield "The Groq request was too large or invalid. Try a shorter question or index fewer sources."
            return

    answer_text = response.content or "I do not know based on the available context."

    # simulate streaming for Streamlit UI
    for word in answer_text.split():
        yield word + " "



def generate_answer(query, context, memory_text):
    return "".join(stream_answer(query, context, memory_text))
