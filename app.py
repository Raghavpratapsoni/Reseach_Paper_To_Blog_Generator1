import os
import streamlit as st
import fitz  # PyMuPDF
import tempfile
from huggingface_hub import login
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS


# 🔐 API TOKENS (HuggingFace token is hardcoded as requested)
HUGGINGFACE_TOKEN = "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"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Warn if tokens are missing
if not HUGGINGFACE_TOKEN or not GROQ_API_KEY:
    st.warning("Please set the HUGGINGFACE_TOKEN and GROQ_API_KEY environment variables.")

# Authenticate Hugging Face & Groq
if HUGGINGFACE_TOKEN:
    login(HUGGINGFACE_TOKEN)
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Extract text from PDF
def get_pdf_text(pdf_file_path):
    text = ""
    try:
        doc = fitz.open(pdf_file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

# Split into chunks
def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

# Create vectorstore using sentence-transformers
def get_vectorstore(chunks):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return FAISS.from_texts(texts=chunks, embedding=embeddings)
    except Exception as e:
        st.error(
            "Failed to load embedding model. Please check your HuggingFace token and internet connection.\n"
            f"Error: {e}"
        )
        return None

# Blog-style prompt template
template = """
You are an expert science communicator. Based on the content of the research paper below, write a blog-style explainer suitable for a non-expert audience.

Follow this format:

Title: (Make it catchy and easy to understand)

1. The Problem
- What problem does the paper address?
- Why is it important?

2. The Approach
- What method or idea is proposed?
- How does it work in simple terms?

3. Key Takeaways
- What are the key findings?
- How can this be useful?

Write in a conversational tone, use analogies, and avoid technical jargon. The goal is to help general readers understand and appreciate the research.

Context: {context}
Research Paper Text: {question}

Generate a blog-style summary below:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# Streamlit App UI
st.set_page_config(page_title="🧠 Research Paper Explainer", layout="centered")
st.title("📄 Research Paper to Blog Converter")
st.write("Upload a research paper PDF, and get a blog-style summary for general readers.")

pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

if pdf_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name

        with st.spinner("🔍 Extracting text from PDF..."):
            raw_text = get_pdf_text(tmp_path)

        if not raw_text:
            st.error("No text could be extracted from the PDF.")
        else:
            st.success(f"✅ Extracted {len(raw_text)} characters from the PDF.")

            with st.spinner("🔗 Splitting and indexing text..."):
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)

            if vectorstore is not None:
                st.success("✅ Vectorstore created successfully.")

                with st.spinner("🤖 Generating blog-style summary..."):
                    try:
                        llm = ChatGroq(
                            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
                            temperature=0.7,
                            request_timeout=30
                        )
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            retriever=vectorstore.as_retriever(),
                            chain_type_kwargs={"prompt": prompt}
                        )

                        response = qa_chain.invoke({"query": raw_text})
                        blog_summary = response["result"]
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
                        blog_summary = None

                if blog_summary:
                    st.subheader("📝 Blog Summary")
                    st.write(blog_summary)
                    st.download_button("📥 Download Summary", blog_summary, file_name="summary.txt", mime="text/plain")
    except Exception as e:
        st.error(f"An error occurred: {e}")
