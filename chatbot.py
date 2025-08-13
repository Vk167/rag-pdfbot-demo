import os
import shutil
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.docstore.document import Document
# from langchain.embeddings import HuggingFaceEmbeddings
# # from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from difflib import get_close_matches
from form_mapping import form_mapping

from utils import *

# Main chat function
def run_chatbot():
    print("Loading PDFs and vector store...")
    # text = load_pdfs(PDF_DIR)
    # chunks = split_text(text)
    # update_faiss_index(chunks)
    update_faiss_index_if_needed()

    # vectorstore = FAISS.load_local(INDEX_DIR, get_embedding_model(),)
    vectorstore = FAISS.load_local(INDEX_DIR, get_embedding_model(), allow_dangerous_deserialization=True)

    retriever = vectorstore.as_retriever()
    llm = init_llm()
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print("\n Chatbot is ready. Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit", "stop"]:
            break

        form_response = match_form(query)
        if form_response:
            print(f"Bot: {form_response}\n")
            continue

        response = chain.run(query)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    run_chatbot()
