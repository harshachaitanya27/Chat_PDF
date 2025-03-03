# from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma # type: ignore
from langchain.schema.document import Document
import dotenv
import os
import shutil
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()

def main():
    
    documents = load_documents()
    chunks = split_text(documents)
    add_data_to_db(chunks)
    
def get_embeddings():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return embeddings

if not os.environ.get('OPENAI_API_KEY'):
    print('Please give API key')

DATA_PATH = 'data/'
CHROMA_PATH = 'chroma'

def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex=False,
    )
    
    return text_splitter.split_documents(documents)

def add_data_to_db(chunks: list[Document]):
    
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function = get_embeddings()
    )
    
    exisiting_items = db.get(include=[])
    exisiting_ids = exisiting_items['ids']
    
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    print(f"Number of exisiting documents in db: {len(exisiting_ids)}")
    
    new_chunks = []
    for chunk in chunks:
        if chunk in chunks_with_ids:
            if chunk.metadata['id'] not in exisiting_ids:
                new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"Added New Chunks: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata['id'] for chunk in new_chunks]
    
        db.add_documents(new_chunks, ids = new_chunk_ids)
        #db.persist()
    else:
        print("No new chunks added!")

def calculate_chunk_ids(chunks):
    
    
    last_page_id = None
    current_chunk_index = 0
    
    for chunk in chunks:
        source = chunk.metadata.get('source')
        page = chunk.metadata.get('page')
        current_page_id = f'{source}:{page}'
        
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        current_chunk_id = f"{current_page_id}:{current_chunk_index}"
        
        last_page_id = current_page_id
        
        chunk.metadata['id'] = current_chunk_id
        
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == '__main__':
    main()
        
        
        
    


