import os
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.docstore.document import Document
# from langchain.llms import LlamaCpp
from difflib import get_close_matches

from form_mapping import form_mapping

PDF_DIR = "data/pdfs"
INDEX_DIR = "faiss_index"
# EMBEDDING = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
MODEL_PATH = r"D:\PDFChatbotNew\model\mistral-7b-instruct-v0.1.Q4_K_M.gguf"
INDEX_TRACK_FILE = "indexed_files.json"


# Load and chunk PDFs
def load_pdfs(pdf_dir):
    text = ""
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            path = os.path.join(pdf_dir, file)
            reader = PdfReader(path)
            for page in reader.pages:
                text += page.extract_text() or ""
    return text.strip()

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

# Initialize embedding model
# def get_embedding_model():
#     return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

from functools import lru_cache

@lru_cache(maxsize=1)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Build or update FAISS index
# def update_faiss_index(text_chunks, index_dir=INDEX_DIR):
#     embeddings = get_embedding_model()
#     docs = [Document(page_content=chunk) for chunk in text_chunks]
#
#     # Load if exists
#     if os.path.exists(index_dir):
#         index = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
#         index.add_documents(docs)
#     else:
#         index = FAISS.from_documents(docs, embeddings)
#
#     index.save_local(index_dir)
#     return index

# Initialize LLM

def get_indexed_files():
    if os.path.exists(INDEX_TRACK_FILE):
        with open(INDEX_TRACK_FILE, "r") as f:
            return json.load(f)
    return {}

def save_indexed_files(data):
    with open(INDEX_TRACK_FILE, "w") as f:
        json.dump(data, f)

def update_faiss_index_if_needed(pdf_dir=PDF_DIR, index_dir=INDEX_DIR):
    indexed_files = get_indexed_files()
    new_docs = []
    current_files = {}

    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            path = os.path.join(pdf_dir, file)
            mtime = os.path.getmtime(path)
            current_files[file] = mtime

            if file not in indexed_files or indexed_files[file] != mtime:
                print(f"Indexing new or updated file: {file}")
                reader = PdfReader(path)
                text = "".join(page.extract_text() or "" for page in reader.pages)
                chunks = split_text(text)
                docs = [Document(page_content=chunk) for chunk in chunks]
                new_docs.extend(docs)

    if new_docs:
        embeddings = get_embedding_model()
        if os.path.exists(index_dir):
            index = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
            index.add_documents(new_docs)
        else:
            index = FAISS.from_documents(new_docs, embeddings)
        index.save_local(index_dir)
        save_indexed_files(current_files)
        print(f"✅ Added {len(new_docs)} new chunks to FAISS index.")
        return True
    else:
        print("✅ No new or modified PDFs detected. FAISS index is up-to-date.")
        return False
    # if new_docs:
    #     embeddings = get_embedding_model()
    #     if os.path.exists(index_dir):
    #         index = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    #         index.add_documents(new_docs)
    #     else:
    #         index = FAISS.from_documents(new_docs, embeddings)
    #     index.save_local(index_dir)
    #     save_indexed_files(current_files)
    #     print(f"✅ Added {len(new_docs)} new chunks to FAISS index.")
    # else:
    #     print("✅ No new or modified PDFs detected. FAISS index is up-to-date.")


def init_llm():
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=4,
        temperature=0.1
    )

# Match forms via fuzzy logic
def match_form(query):
    query = query.lower()
    for key in form_mapping:
        if key in query:
            return f"📄 Here is the **{key.title()}**:\n{form_mapping[key]}"
    match = get_close_matches(query, form_mapping.keys(), n=1, cutoff=0.6)
    if match:
        matched = match[0]
        return f"📄 Here is the **{matched.title()}**:\n{form_mapping[matched]}"
    return None


