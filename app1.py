import streamlit as st
import os
from io import BytesIO
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain


# ----------------------- Helper Functions -----------------------

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def create_faiss_vector_store(text, path="faiss_index"):
    """Creates a FAISS vector store from the given text."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(path)


def load_faiss_vector_store(path="faiss_index"):
    """Loads the FAISS vector store from local path."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vector_store


def build_qa_chain(vector_store_path="faiss_index"):
    """Builds the question answering chain using Ollama + FAISS retriever."""
    vector_store = load_faiss_vector_store(vector_store_path)
    retriever = vector_store.as_retriever()
    llm = Ollama(model="llama3:instruct")  # Change model tag if needed
    doc_chain = load_qa_chain(llm, chain_type="stuff")
    qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=doc_chain)
    return qa_chain


# ----------------------- Streamlit UI -----------------------


st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📄🧠 RAG Chatbot with FAISS + LLaMA")
st.write("Upload a PDF and ask questions based on its content.")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:

    # Read file directly from memory
    pdf_bytes = uploaded_file.read()
    pdf_buffer = BytesIO(pdf_bytes)

    # Extract and process PDF, no saving to disk
    st.info("Extracting text and creating vector store...")
    text = extract_text_from_pdf(pdf_buffer)

    # Directly create FAISS embeddings from extracted text
    create_faiss_vector_store(text)

    # Load QA chain
    st.info("Initializing LLaMA chatbot...")
    qa_chain = build_qa_chain()
    st.success("✅ Chatbot is ready! Ask your question below.")

    # Ask question
    question = st.text_input("❓ Ask a question about the uploaded PDF:")
    
    if question:
        st.info("🔍 Searching for the answer...")
        answer = qa_chain.invoke({"query": question})

        # Display input key info and result
        st.markdown("### 📥 Expected Input Keys")
        st.code(f"{qa_chain.input_keys}", language="python")

        st.markdown("### 💬 Answer")
        st.success(answer['result'])
