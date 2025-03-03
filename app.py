import streamlit as st
import os
import shutil
from load_data import load_documents, split_text, add_data_to_db, clear_database, get_embeddings
from chat_pdf import pdf_rag
from langchain_community.vectorstores import Chroma

# Paths
DATA_PATH = "data/"
CHROMA_PATH = "chroma"

# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)

# Streamlit App
st.title("📄 AI-Powered PDF Q&A")

# File Upload UI
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(DATA_PATH, file.name), "wb") as f:
            f.write(file.getbuffer())
    st.success(f"✅ Uploaded {len(uploaded_files)} files.")

    if st.button("Process PDFs"):
        with st.spinner("Processing PDFs..."):
            documents = load_documents()
            chunks = split_text(documents)
            add_data_to_db(chunks)
            st.success(f"✅ Indexed {len(chunks)} chunks.")

# Query UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.subheader("🔍 Ask AI About Your PDFs")
query = st.text_input("Enter your question")

if query:
    with st.spinner("Fetching answer..."):
        response = pdf_rag(query)
        st.markdown(response)
        st.session_state.chat_history.append({"query": query, "response": response})


# Clear Database Button
if st.button("🗑️ Delete All PDFs & Clear Data"):
    clear_database()
    shutil.rmtree(DATA_PATH)
    os.makedirs(DATA_PATH)  # Recreate empty folder
    st.success("✅ All PDFs and indexed data deleted!")

st.subheader("Chat History")
for entry in st.session_state.chat_history[::-1]:  # Show latest first
    st.markdown(f"**Q:** {entry['query']}")
    st.markdown(f"**A:** {entry['response']}")
    st.markdown("---")
