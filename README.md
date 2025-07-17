#  RAG Chatbot with FAISS, LangChain, and LLaMA

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and interactively ask questions about their content. The system leverages state-of-the-art language models, vector search, and modern web technologies to provide accurate, context-aware answers from your documents.

---

## 🚀 Features
- **PDF Upload & Parsing:** Upload any PDF and extract its text for analysis.
- **Semantic Search:** Uses FAISS and HuggingFace embeddings for fast, relevant document retrieval.
- **Conversational QA:** Ask questions about your uploaded PDF and get precise answers using LLaMA via Ollama and LangChain.
- **Modern Web UI:** Built with Streamlit for an intuitive, interactive experience.

---

## 📁 Project Structure
```
RAG/
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── faiss_index/          # Directory for FAISS vector store files
│   ├── index.faiss       # FAISS index data
│   └── index.pkl         # Metadata for the vector store
├── uploaded/             # Uploaded PDF files
│   ├── xxx.pdf
│   └── xxx.pdf
```

---

## 🛠️ Technologies Used
- **Python:** Core programming language.
- **Streamlit:** For building the interactive web interface.
- **PyPDF2:** Extracts text from PDF files.
- **LangChain:** Orchestrates the RAG pipeline and integrates LLMs.
- **HuggingFace Transformers:** Provides sentence embeddings for semantic search.
- **FAISS:** Efficient similarity search and vector storage.
- **Ollama:** Runs the LLaMA language model locally for question answering.

### Functionality Overview
- **PDF Parsing:** `PyPDF2` reads and extracts text from uploaded PDFs.
- **Text Chunking:** `LangChain` splits text into manageable chunks for embedding.
- **Embedding & Indexing:** `HuggingFaceEmbeddings` creates vector representations; `FAISS` stores and retrieves them.
- **Retrieval QA:** `LangChain`'s `RetrievalQA` chain fetches relevant chunks and generates answers using LLaMA via `Ollama`.
- **Web App:** `Streamlit` provides file upload, chat interface, and real-time feedback.

---

## ⚡ Getting Started
1. **Clone the repository:**
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot
2. **Install dependencies:**
   pip install -r requirements.txt
3. **Run the app:**
   streamlit run app.py
4. **Open in browser:**
   Visit [http://localhost:8501](http://localhost:8501) to use the chatbot.

---

## 📝 Example Usage
1. Upload a PDF file using the sidebar.
2. Wait for the system to process and index the document.
3. Type your question in the chat box (e.g., "What is the main topic of this document?").
4. Receive an answer generated from the content of your PDF.

---

## 🔗 Links
- **GitHub Repository:** [https://github.com/yourusername/rag-chatbot](https://github.com/yourusername/rag-chatbot)
- **Live Demo:** [https://your-demo-link.com](https://your-demo-link.com)

---

## 📚 More Information
- **Extensible:** Easily swap out the LLM, embeddings, or vector store for your needs.
- **Local & Private:** All processing is done locally; your documents never leave your machine.
- **Customizable:** Adapt the chunk size, overlap, or model for different document types.

---

## 📄 License
This project is licensed under the MIT License. 
