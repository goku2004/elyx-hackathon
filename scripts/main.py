# main.py
# To run this:
# 1. pip install fastapi uvicorn google-generativeai python-dotenv "uvicorn[standard]"
# 2. Create a file named .env in the same directory and add your API key:
#    GEMINI_API_KEY="YOUR_API_KEY_HERE"
# 3. In your terminal, run: uvicorn main:app --reload

import os
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
genai.configure(api_key=api_key)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS (Cross-Origin Resource Sharing) to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request Body ---
class ChatPart(BaseModel):
    from_role: str
    text: str

class GenerationRequest(BaseModel):
    role: str
    prompt: str
    history: List[Dict[str, Any]]
    persona: str

# --- API Endpoint ---
@app.post("/generate")
async def generate_conversation_turn(request: GenerationRequest):
    """
    Receives a prompt and history from the frontend, calls the Gemini API,
    and returns the generated text.
    """
    try:
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash-latest',
            system_instruction=request.persona
        )

        # Convert frontend history to the format expected by the Gemini API
        chat_history = []
        for h in request.history:
            role = 'user' if h.get('from') == 'Rohan' else 'model'
            chat_history.append({'role': role, 'parts': [f"({h.get('from')}): {h.get('text')}"]})
        
        # The new prompt from the user
        full_prompt = f"({request.role}): {request.prompt}"

        # Generate content
        response = model.generate_content(
            [*chat_history, {'role': 'user', 'parts': [full_prompt]}],
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=150
            )
        )
        
        return {"text": response.text}

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"text": f"[Backend Error: {str(e)}"}

@app.get("/")
def read_root():
    return {"message": "Elyx Journey Simulator Backend is running."}

