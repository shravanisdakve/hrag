from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import shutil
import os
from .rag_engine import index_document, query_knowledge_base

app = FastAPI(title="PrivatAI Hospital Server")

# Create a temp folder for uploads
os.makedirs("temp_uploads", exist_ok=True)

@app.post("/upload/")
async def upload_document(
    file: UploadFile = File(...), 
    department: str = Form(...)
):
    """
    Endpoint for Admin to upload documents to a specific department.
    """
    file_path = f"temp_uploads/{file.filename}"
    
    # Save file locally first
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Pass to RAG Engine for Indexing
    try:
        message = index_document(file_path, department)
        os.remove(file_path) # Cleanup
        return {"status": "success", "message": message}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/chat/")
async def chat(
    query: str = Form(...), 
    department: str = Form(...)
):
    """
    Endpoint for Users to ask questions restricted to their department.
    """
    try:
        answer = query_knowledge_base(query, department)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}


