import os
import streamlit as st
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]



def build_messages(query, context, memory_text):
    system_prompt = f"""
You are a helpful AI assistant.

Use the provided context to answer the question.

Context:
{context}

Conversation History:
{memory_text}

If the answer is not found in the context, say you don't know.
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]



def stream_answer(query, context, memory_text):
    if "GROQ_API_KEY" not in st.secrets:
        yield "GROQ_API_KEY is missing in Streamlit secrets."
        return

    if not context.strip():
        yield "I could not find enough evidence to answer that question yet."
        return

    messages = build_messages(query, context, memory_text)

    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.2
    )

    response = llm.invoke(messages)

    answer_text = response.content

    # simulate streaming
    for word in answer_text.split():
        yield word + " "



def generate_answer(query, context, memory_text):
    return "".join(stream_answer(query, context, memory_text))
