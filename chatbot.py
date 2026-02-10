from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_db"

embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ✅ UPDATED MODEL NAME
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    streaming=True,
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

def stream_response(message, history):
    docs = retriever.invoke(message)

    knowledge = "\n\n".join(doc.page_content for doc in docs)

    rag_prompt = f"""
You are an assistant that answers ONLY using the knowledge below.
Do NOT use outside knowledge.
Do NOT mention the knowledge source.

Question:
{message}

Knowledge:
{knowledge}
"""

    partial = ""
    for chunk in llm.stream(rag_prompt):
        partial += chunk.content
        yield partial

chatbot = gr.ChatInterface(
    stream_response,
    textbox=gr.Textbox(
        placeholder="Ask a question...",
        container=False,
        autoscroll=True,
        scale=7,
    ),
)

chatbot.launch()
