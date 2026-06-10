import os
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from phase2_chatbot.retriever import retrieve_relevant_chunks
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

SYSTEM_PROMPT = """You are a helpful AI tutor trained on Campus X educational content about AI, Machine Learning, and Deep Learning.

Answer the user's question using ONLY the provided context from Campus X video transcripts.
If the context doesn't contain enough information, say so honestly.
Always be clear, structured, and beginner-friendly in your explanations.
Mention the video source when relevant."""


def answer_question(question: str, chat_history: list = []) -> dict:
    """Retrieve context and generate answer using RAG."""
    chunks = retrieve_relevant_chunks(question, top_k=5)

    context = "\n\n".join([
        f"[From: {c['title']}]\n{c['text']}"
        for c in chunks
    ])

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *chat_history,
        HumanMessage(content=f"""Context from Campus X videos:
{context}

Question: {question}""")
    ]

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": [{"title": c["title"], "url": c["url"]} for c in chunks]
    }
