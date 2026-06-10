import streamlit as st
from phase2_chatbot.rag_chain import answer_question
from langchain.schema import HumanMessage, AIMessage

st.set_page_config(page_title="Campus X RAG Tutor", page_icon="🎓", layout="wide")
st.title("🎓 Campus X AI Tutor")
st.caption("Ask anything about AI, ML, DL — powered by Campus X video transcripts")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "lc_history" not in st.session_state:
    st.session_state.lc_history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- [{src['title']}]({src['url']})")

if prompt := st.chat_input("Ask a question about AI/ML/DL..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer_question(prompt, st.session_state.lc_history)
            answer = result["answer"]
            sources = result["sources"]

        st.markdown(answer)
        if sources:
            with st.expander("📚 Sources"):
                for src in sources:
                    st.markdown(f"- [{src['title']}]({src['url']})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    st.session_state.lc_history.extend([
        HumanMessage(content=prompt),
        AIMessage(content=answer)
    ])
