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

def main():
    
    while True:
        query = input("What would you like to know? (Type exit to exit)\n")
        if query.lower() == 'exit':
            return
        pdf_rag(query)
        


def pdf_rag(query: str):
    db = Chroma(
        persist_directory = CHROMA_PATH, embedding_function = get_embeddings()
    )
    
    results = db.similarity_search_with_score(query, k = 2)
    
    context_text = "\n\n-------------\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = context_text, question = query)
    
    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get('id',None) for doc,_score in results]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    print(formatted_response)
    return formatted_response

if __name__=="__main__":
    main()
    
    