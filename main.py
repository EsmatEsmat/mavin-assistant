import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
import pandas as pd

# Setup Logging
logging.basicConfig(filename='chat_logs.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Data and Model with 'latin1' encoding to handle special characters
faq = pd.read_csv("master_faq.csv", encoding='latin1')

model = SentenceTransformer('all-MiniLM-L6-v2')
# Using the exact header names from your screenshot
faq_embs = model.encode(faq["Main Question"].tolist(), convert_to_tensor=True)

class QueryRequest(BaseModel):
    message: str

def is_urdu(text):
    return any(u'\u0600' <= char <= u'\u06FF' for char in text)

@app.post("/chat")
async def chat(request: QueryRequest):
    try:
        prompt = request.message
        urdu = is_urdu(prompt)
        logging.info(f"Query: {prompt}")

        # Greeting Logic
        greetings_ur = ["سلام", "آداب", "ہیلو"]
        greetings_en = ["hi", "hello", "hey"]
        
        if any(g in prompt.lower() for g in greetings_ur):
            ans = "آداب! میں MAVIN معاوِن ہوں۔ میں آپ کی کس طرح مدد کر سکتا ہوں؟"
        elif any(g in prompt.lower() for g in greetings_en):
            ans = "Adaab! I am MAVIN معاوِن. How can I help you?"
        else:
            # AI Logic
            q = GoogleTranslator(source='ur', target='en').translate(prompt) if urdu else prompt
            val, idx = util.cos_sim(model.encode(q, convert_to_tensor=True), faq_embs).max(dim=1)
            
            if val > 0.4:
                # Using the exact "Answer" header from your screenshot
                ans = faq.iloc[int(idx)]["Answer"]
                if urdu: ans = GoogleTranslator(source='en', target='ur').translate(ans)
            else:
                ans = "معذرت، میں صرف مانو CDOE سے متعلق سوالات دے سکتا ہوں۔" if urdu else "I can only answer MANUU CDOE related questions."
        
        logging.info(f"Response: {ans}")
        return {"response": ans}
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Error occurred while processing.")