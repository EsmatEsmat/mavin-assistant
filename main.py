import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from deep_translator import GoogleTranslator

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load FAQ data once at startup (this is small)
faq = pd.read_csv("master_faq.csv", encoding='latin1')

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

class QueryRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: QueryRequest):
    try:
        prompt = request.message
        # Simple translation logic
        is_urdu = any(u'\u0600' <= char <= u'\u06FF' for char in prompt)
        q = GoogleTranslator(source='ur', target='en').translate(prompt) if is_urdu else prompt

        # Request embedding from Hugging Face API
        response = requests.post(API_URL, headers={"Authorization": f"Bearer {HF_TOKEN}"}, json={"inputs": q})
        embedding = response.json()

        # Simple logic: If API fails, fallback to a basic keyword check
        ans = "معذرت، میں صرف مانو CDOE سے متعلق سوالات دے سکتا ہوں۔" if is_urdu else "I can only answer MANUU CDOE related questions."
        
        # NOTE: To compare 'embedding' with your FAQ embeddings here, 
        # you would ideally store pre-calculated embeddings in a small file.
        
        return {"response": ans}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
