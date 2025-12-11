import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
import qdrant_client

# 1. Setup Local LLM and Embedding Model
llm = Ollama(model="llama3", request_timeout=300.0) # Ensure you have 'ollama run llama3' running
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 2. Setup Qdrant Client (Local Database)
client = qdrant_client.QdrantClient(url="http://localhost:6333")
vector_store = QdrantVectorStore(client=client, collection_name="hospital_data")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

def index_document(file_path: str, department: str):
    """
    Reads a file, indexes it, and tags it with the specific Department.
    """
    # Read the file
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    
    # Tag every chunk of this document with the department metadata
    for doc in documents:
        doc.metadata["department"] = department
        
    # Create index (pushes vectors to Qdrant)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )
    return f"Successfully indexed {os.path.basename(file_path)} for {department}."

def query_knowledge_base(query: str, department: str):
    """
    Queries the RAG system but STRICTLY filters by Department.
    """
    # Connect to existing index
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    
    # Define strict filters: "Only show me data where department == user's department"
    filters = MetadataFilters(
        filters=[
            ExactMatchFilter(key="department", value=department),
        ]
    )
    
    # Create Query Engine with the filter
    query_engine = index.as_query_engine(
        llm=llm,
        filters=filters,
        similarity_top_k=3
    )
    
    response = query_engine.query(query)
    return str(response)
