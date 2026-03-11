import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

api_key = st.secrets.get("GROQ_API_KEY")

if api_key:
    os.environ["GROQ_API_KEY"] = api_key



def _build_messages(query, context, memory_text):
    system_prompt = f"""
You are a helpful AI assistant.

Use the provided context to answer the user's question.

Context:
{context}

Conversation History:
{memory_text}

If the answer is not present in the context, say you do not know.
"""
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]



def stream_answer(query, context, memory_text):
    if not api_key:
        yield "GROQ_API_KEY is missing in Streamlit secrets."
        return

    if not context.strip():
        yield "I could not find enough evidence to answer that question yet."
        return

    messages = _build_messages(query, context, memory_text)

    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.2
    )

    response = llm.invoke(messages)

    answer_text = response.content

    # simulate streaming for Streamlit UI
    for word in answer_text.split():
        yield word + " "



def generate_answer(query, context, memory_text):
    return "".join(stream_answer(query, context, memory_text))
