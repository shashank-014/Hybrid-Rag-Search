from langchain.memory import ConversationBufferMemory



def create_memory() -> ConversationBufferMemory:
    return ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=False,
    )



def load_memory_text(memory: ConversationBufferMemory) -> str:
    values = memory.load_memory_variables({})
    return values.get("chat_history", "")



def save_turn(memory: ConversationBufferMemory, question: str, answer: str) -> None:
    memory.save_context({"question": question}, {"answer": answer})
