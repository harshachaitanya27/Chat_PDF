import streamlit as st
from load_data import get_embeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

CHROMA_PATH = 'chroma'

PROMPT_TEMPLATE = '''

Answer the question based only on the following:
{context}

If {context} does not have any answer, reply:
"The data I have does not have any information regarding:
{question}"

------------------------------------------
Answer the question based on the above context: {question}

'''

# Streamlit App Title
st.title("📄 AI-Powered PDF Q&A")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
query = st.text_input("What would you like to know?", "")

if st.button("Ask"):
    if query:
        with st.spinner("Searching for relevant information..."):
            db = Chroma(
                persist_directory=CHROMA_PATH, embedding_function=get_embeddings()
            )
            
            results = db.similarity_search_with_score(query, k=2)
            
            context_text = "\n\n-------------\n\n".join([doc.page_content for doc, _score in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query)
            
            model = Ollama(model="mistral")
            response_text = model.invoke(prompt)

            sources = [doc.metadata.get('id', None) for doc, _score in results]
            formatted_response = f"**Response:**\n\n{response_text}\n\n**Sources:** {sources}"
            sources = []

            # Store query & response in session state
            st.session_state.chat_history.append({"query": query, "response": formatted_response})

# Display chat history
st.subheader("Chat History")
for entry in st.session_state.chat_history[::-1]:  # Show latest first
    st.markdown(f"**Q:** {entry['query']}")
    st.markdown(f"**A:** {entry['response']}")
    st.markdown("---")
