import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pypdf import PdfReader

# ---- CONFIG ----
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your-groq-api-key-here")
# ---- PAGE SETUP ----
st.set_page_config(page_title="RAG Document Chat", page_icon="📄")
st.title("📄 Chat With Your PDF")
st.write("Upload a PDF and ask anything about it!")

# ---- LOAD PDF ----
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(raw_text)

    with st.spinner("Reading your document..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = FAISS.from_texts(chunks, embeddings)

    st.success("Document ready! Ask your question below.")

    # ---- PROMPT ENGINEERING ----
    prompt = PromptTemplate.from_template("""
    You are a helpful assistant. Use the context below to answer the question.
    If the answer is not in the context, say "I couldn't find that in the document."

    Context: {context}

    Question: {question}

    Answer clearly and in simple English:
    """)

    # ---- SETUP LLM ----
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

    # ---- RAG CHAIN ----
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # ---- CHAT INPUT ----
    question = st.text_input("Ask a question about your PDF:")

    if question:
        with st.spinner("Thinking..."):
            answer = chain.invoke(question)
        st.markdown("### 💬 Answer:")
        st.write(answer)