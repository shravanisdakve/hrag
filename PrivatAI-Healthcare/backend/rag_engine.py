import os
import faiss
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Setup Local LLM and Embedding
llm = Ollama(model="llama3", request_timeout=300.0)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Define where we save the FAISS indexes
STORAGE_BASE_DIR = "./faiss_indexes"
os.makedirs(STORAGE_BASE_DIR, exist_ok=True)

def get_department_path(department: str):
    """Returns the folder path for a specific department's index."""
    # simple cleanup to ensure safe folder names
    clean_dept = "".join(x for x in department if x.isalnum())
    return os.path.join(STORAGE_BASE_DIR, clean_dept)

def index_document(file_path: str, department: str):
    """
    Creates or updates a FAISS index SPECIFIC to that department.
    """
    dept_path = get_department_path(department)
    
    # 1. Load the new document
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    # 2. Check if an index already exists for this department
    if os.path.exists(dept_path):
        # Load existing index
        vector_store = FaissVectorStore.from_persist_dir(dept_path)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=dept_path
        )
        index = load_index_from_storage(storage_context, embed_model=embed_model)
        
        # Add new document to it
        index.insert(documents[0])
        
        # Save it back to disk
        index.storage_context.persist(persist_dir=dept_path)
    else:
        # Create a NEW index from scratch
        # Dimension 384 is for BAAI/bge-small-en-v1.5. 
        # If you change embedding models, change this dimension!
        d = 384 
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context, 
            embed_model=embed_model
        )
        # Save to disk
        index.storage_context.persist(persist_dir=dept_path)

    return f"Securely indexed document into {department} vault."

def query_knowledge_base(query: str, department: str):
    """
    Loads ONLY the specified department's index to answer.
    """
    dept_path = get_department_path(department)
    
    if not os.path.exists(dept_path):
        return "No secure records found for this department yet."

    # Load the specific department index
    vector_store = FaissVectorStore.from_persist_dir(dept_path)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=dept_path
    )
    index = load_index_from_storage(storage_context, embed_model=embed_model)

    # Query
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)
    response = query_engine.query(query)
    
    return str(response)