import streamlit as st
from langchain.chains import RetrievalQA
from form_mapping import form_mapping
# from utils import update_faiss_index_if_needed, get_embedding_model, match_form, init_llm
from langchain_community.vectorstores import FAISS
from utils import *
# INDEX_DIR = "path/to/index"  # make sure this is correct

st.title("📚 RAG Chatbot")

@st.cache_resource
def load_chain():
    """Load the FAISS vector store and create the retrieval chain."""
    update_faiss_index_if_needed()
    vectorstore = FAISS.load_local(INDEX_DIR, get_embedding_model(), allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    llm = init_llm()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

chain = load_chain()

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Input box at bottom of screen
query = st.chat_input("Ask me something about the PDFs...")

if query:
    # Store user message
    st.session_state["messages"].append({"role": "user", "content": query})

    # Check if it's a form match
    form_response = match_form(query)
    if form_response:
        bot_reply = form_response
    else:
        bot_reply = chain.run(query)

    # Store bot reply
    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})

# Display chat messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
