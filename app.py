import os
import streamlit as st
import time
from utils import (
    update_faiss_index_if_needed,
    init_llm,
    get_embedding_model,
    match_form,
)
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

PDF_DIR = "data/pdfs"
INDEX_DIR = "faiss_index"

st.set_page_config(page_title="📘 University Chatbot", layout="wide")
st.title("📘 University Chatbot ")

# === INITIAL SETUP: Store vectorstore and LLM in session ===
if "llm" not in st.session_state:
    st.session_state.llm = init_llm()

if "qa_chain" not in st.session_state:
    st.info("🔁 Initializing chatbot...")
    update_faiss_index_if_needed()  # Initial check
    embeddings = get_embedding_model()
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    st.session_state.qa_chain = RetrievalQA.from_chain_type(llm=st.session_state.llm, retriever=retriever)
    st.success("✅ Chatbot is ready!")

# === CHECK FOR NEW FILES PERIODICALLY ===
# def auto_refresh_index():
#     if "last_check" not in st.session_state:
#         st.session_state.last_check = 0
#
#     # Refresh every 30 seconds (you can adjust this)
#     if time.time() - st.session_state.last_check > 30:
#         st.session_state.last_check = time.time()
#         update_faiss_index_if_needed()
#         # Reload vectorstore if any new files are added
#         embeddings = get_embedding_model()
#         vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
#         retriever = vectorstore.as_retriever()
#         st.session_state.qa_chain = RetrievalQA.from_chain_type(
#             llm=st.session_state.llm,
#             retriever=retriever,
#         )

##RUNNING CODE
def auto_refresh_index():
    if "last_check" not in st.session_state:
        st.session_state.last_check = 0

    if time.time() - st.session_state.last_check > 30:
        st.session_state.last_check = time.time()
        update_faiss_index_if_needed()
        # Now reload FAISS and update retriever + chain
        embeddings = get_embedding_model()
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            retriever=retriever,
        )
        # st.toast("🔄 FAISS index updated with new PDF(s)!")

##To be tested
# def auto_refresh_index():
#     last_check = st.session_state.get("last_check", 0)
#
#     if time.time() - last_check > 30:
#         st.session_state.last_check = time.time()
#         print(f"[{time.strftime('%H:%M:%S')}] Checking for PDF updates...")
#
#         updated = update_faiss_index_if_needed()
#         if updated:
#             embeddings = get_embedding_model()
#             vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
#             retriever = vectorstore.as_retriever()
#             st.session_state.qa_chain = RetrievalQA.from_chain_type(
#                 llm=st.session_state.llm,
#                 retriever=retriever,
#             )
#             st.toast("🔄 FAISS index updated with new PDF(s)!", icon="📚")


auto_refresh_index()

st.write(f"⏰ Last refreshed at: {time.strftime('%H:%M:%S')}")

# === INITIALIZE CHAT HISTORY BEFORE USE ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === CHAT INTERFACE ===
st.subheader("💬 Ask your question about the PDFs")
with st.form("chat_form"):
    query = st.text_input("Type your question here:")
    submitted = st.form_submit_button("Submit")


if submitted and query:
    form_response = match_form(query)
    # if form_response:
    #     st.info(form_response)
    if form_response:
        st.session_state.chat_history.insert(0, ("Bot", form_response))
        st.session_state.chat_history.insert(0, ("User", query))
    else:
        with st.spinner("🤔 Thinking..."):
            response = st.session_state.qa_chain.run(query)
        st.session_state.chat_history.insert(0, ("Bot", response))
        st.session_state.chat_history.insert(0, ("User", query))
        # st.success(response)

# === DISPLAY CHAT HISTORY (Latest First) ===
for role, message in st.session_state.chat_history:
    if message:
        st.markdown(f"**{role}:** {message}")

# === Periodic refresh every 30 seconds ===
REFRESH_INTERVAL_SEC = 30
st.markdown(
    f"""
    <script>
        function refresh() {{
            window.location.reload();
        }}
        setTimeout(refresh, {REFRESH_INTERVAL_SEC * 1000});
    </script>
    """,
    unsafe_allow_html=True,
)
