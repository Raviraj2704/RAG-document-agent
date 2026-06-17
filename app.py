import streamlit as st
from groq import Groq
import fitz
import os
import docx
import pandas as pd
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text_from_file(uploaded_file):
    name = uploaded_file.name.lower()
    text = ""
    source = uploaded_file.name

    if name.endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page_num, page in enumerate(doc):
            text += f"\n[Source: {source} | Page {page_num+1}]\n"
            text += page.get_text()

    elif name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
        text = f"[Source: {source}]\n" + text

    elif name.endswith(".txt"):
        text = f"[Source: {source}]\n" + uploaded_file.read().decode("utf-8")

    elif name.endswith(".xlsx") or name.endswith(".csv"):
        if name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        text = f"[Source: {source}]\n" + df.to_string()

    return text

def create_vectorstore(all_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(all_text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def ask_question(vectorstore, question, chat_history):
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n".join([doc.page_content for doc in docs])

    history_text = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are a helpful assistant. Answer based ONLY on the document context.
Always mention the source file and page number when available.
If answer not found, say "I cannot find this in the uploaded documents."

Previous Conversation:
{history_text}

Document Context:
{context}

Current Question: {question}

Answer (mention source and page):"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )
    return response.choices[0].message.content

# --- UI ---
st.set_page_config(page_title="RAG Document Agent", page_icon="📚", layout="centered")
st.title("📚 RAG Document Agent")
st.markdown("Upload multiple documents → Ask questions → Get answers with sources!")
st.divider()

uploaded_files = st.file_uploader(
    "📎 Upload Documents (PDF, DOCX, TXT, XLSX, CSV)",
    type=["pdf", "docx", "txt", "xlsx", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner(f"📖 Processing {len(uploaded_files)} document(s)..."):
        all_text = ""
        for file in uploaded_files:
            all_text += extract_text_from_file(file)

        vectorstore = create_vectorstore(all_text)

    st.success(f"✅ {len(uploaded_files)} document(s) processed! Ask me anything.")

    with st.expander("📄 Uploaded Files"):
        for file in uploaded_files:
            st.write(f"✅ {file.name}")

    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_question(vectorstore, question, st.session_state.messages)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()