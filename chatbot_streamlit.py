import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
CHROMA_PATH = "chroma_db"

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Chatbot")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_vectorstore(embeddings):
    return Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.5,
        streaming=True,
    )

with st.spinner("Loading..."):
    embeddings = load_embeddings()
    vector_store = load_vectorstore(embeddings)
    llm = load_llm()

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask a question..."):
    st.session_state["messages"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        partial_response = ""

        docs = retriever.invoke(user_input)
        knowledge = "\n\n".join(doc.page_content for doc in docs)

        rag_prompt = f"""
You are an assistant that answers ONLY using the knowledge below.
Do NOT use outside knowledge.
Do NOT mention the knowledge source.

Question:
{user_input}

Knowledge:
{knowledge}
"""

        for chunk in llm.stream(rag_prompt):
            partial_response += chunk.content
            response_placeholder.markdown(partial_response)

    st.session_state["messages"].append(
        {"role": "assistant", "content": partial_response}
    )
