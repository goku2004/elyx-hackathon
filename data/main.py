# main.py
# To run this:
# 1. pip install fastapi uvicorn google-generativeai python-dotenv "uvicorn[standard]" numpy python-dateutil
# 2. Create a file named .env in the same directory and add your API key:
#    GEMINI_API_KEY="YOUR_API_KEY_HERE"
# 3. In your terminal, navigate to this folder and run: uvicorn main:app --reload

import os
import time
import asyncio
import uuid
from datetime import datetime, timedelta, date
from collections import deque
import numpy as np
from dateutil.relativedelta import relativedelta
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=api_key)

# --- Rate Limiter (with bug fix) ---
class RateLimiter:
    def __init__(self, requests_per_minute: int = 15, daily_request_limit: int = 200):
        self.requests_per_minute = requests_per_minute
        self.daily_request_limit = daily_request_limit
        self.request_times = deque()
        self.daily_requests = 0
        self.last_request_date = date.today()
        self.lock = asyncio.Lock()

    async def wait_if_needed(self):
        async with self.lock:
            today = date.today()
            if today != self.last_request_date:
                self.daily_requests, self.last_request_date = 0, today
            if self.daily_requests >= self.daily_request_limit:
                raise HTTPException(status_code=429, detail="Daily request limit exceeded.")
            
            now = time.monotonic()
            while self.request_times and self.request_times[0] <= now - 60:
                self.request_times.popleft()
            
            if len(self.request_times) >= self.requests_per_minute:
                time_since_oldest_request = now - self.request_times[0]
                wait_time = 60 - time_since_oldest_request
                if wait_time > 0:
                    print(f"⏳ Rate limit hit. Waiting for {wait_time:.2f} seconds.")
                    await asyncio.sleep(wait_time)
            
            self.request_times.append(time.monotonic())
            self.daily_requests += 1
            print(f"Request {self.daily_requests}/{self.daily_request_limit} for today.")

rate_limiter = RateLimiter()

# --- Event Generation Logic (Updated) ---
MIN_EVENTS_PER_WEEK = 4
MAX_EVENTS_PER_WEEK = 7
TRAVEL_EVERY_N_WEEKS = 4
EXERCISE_UPDATE_EVERY_WEEKS = 2
ROLE_MEMBER = "Rohan"

MEMBER_TOPICS = [
    {"topic": "garmin_hrv_low", "details": "Low HRV reading on Garmin watch", "expert": "Advik"},
    {"topic": "poor_sleep_score", "details": "Concern about a low sleep score", "expert": "Advik"},
    {"topic": "nutrition_question", "details": "Question about a specific food or diet plan", "expert": "Carla"},
    {"topic": "exercise_soreness", "details": "Unusual muscle soreness after a workout", "expert": "Rachel"},
    {"topic": "symptom_check", "details": "Minor symptom check, e.g., headache or fatigue", "expert": "Dr. Warren"},
    {"topic": "curiosity_longevity", "details": "A general question about a health or longevity topic", "expert": "Neel"}
]

def make_id(prefix): return f"{prefix}-{uuid.uuid4().hex[:8]}"
def iso(dt): return dt.isoformat()

def generate_timeline(signup_date: datetime, months=1):
    timeline = []
    # --- FIX: Loop for exactly 4 weeks for a 1-month simulation ---
    total_weeks = 4 * months
    cur = signup_date

    for week_index in range(total_weeks):
        week_start = cur
        weekly_events = []
        
        # --- FIX: Use week_index + 1 for more natural 1-based counting ---
        # Travel week (happens in the 4th week of the month)
        if (week_index + 1) % TRAVEL_EVERY_N_WEEKS == 0:
            weekly_events.append({"id": make_id("event"), "start_ts": iso(week_start), "event_type": "travel_week", "initiator": ROLE_MEMBER, "metadata": {"note": "Member is traveling"}})
        
        # Exercise update (happens in the 2nd and 4th weeks)
        if (week_index + 1) % EXERCISE_UPDATE_EVERY_WEEKS == 0:
            weekly_events.append({"id": make_id("event"), "start_ts": iso(week_start + timedelta(days=1, hours=9)), "event_type": "exercise_update", "initiator": "Rachel", "metadata": {"details": "Biweekly exercise plan update"}})
        
        target_events_for_week = np.random.randint(MIN_EVENTS_PER_WEEK, MAX_EVENTS_PER_WEEK + 1)
        member_events_to_add = max(0, target_events_for_week - len(weekly_events))

        for _ in range(member_events_to_add):
            ts = week_start + timedelta(days=np.random.randint(0, 6), hours=np.random.randint(7, 22))
            topic_index = np.random.randint(0, len(MEMBER_TOPICS))
            chosen_topic = MEMBER_TOPICS[topic_index]
            weekly_events.append({"id": make_id("event"), "start_ts": iso(ts), "event_type": "member_message", "initiator": ROLE_MEMBER, "metadata": chosen_topic})
            
        timeline.extend(weekly_events)
        cur += timedelta(days=7)

    timeline.sort(key=lambda e: e['start_ts'])
    return timeline

# --- Personas (Updated for WhatsApp style) ---
ELYX_PERSONAS = {
    "Ruby": "You are Ruby, the Concierge. Your role is logistics and scheduling. Your voice is empathetic and organized. Keep your messages very brief and friendly, like a WhatsApp message.",
    "Dr. Warren": "You are Dr. Warren, the Medical Strategist. You interpret labs and set medical direction. Your voice is authoritative and precise. Keep your messages very brief and to the point, like a WhatsApp message.",
    "Advik": "You are Advik, the Performance Scientist. You analyze wearable data. Your voice is analytical and data-driven. Keep your messages very brief and insightful, like a WhatsApp message.",
    "Carla": "You are Carla, the Nutritionist. You design nutrition plans. Your voice is practical and educational. Keep your messages very brief and actionable, like a WhatsApp message.",
    "Rachel": "You are Rachel, the PT/Physiotherapist. You manage exercise programming. Your voice is direct and encouraging. Keep your messages very brief, like a WhatsApp message.",
    "Neel": "You are Neel, the Concierge Lead. You handle strategic reviews. Your voice is reassuring and focused on the big picture. Keep your messages very brief, like a WhatsApp message.",
    "Rohan": "You are Rohan, a 46-year-old busy FinTech executive. You are analytical and direct. You write very short, WhatsApp-style messages."
}

# --- FastAPI App ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class SimulationRequest(BaseModel):
    months: int = 1
    known_issues: Dict[str, bool] = {}

class SimulationResponse(BaseModel):
    event: Dict[str, Any]
    conversation: List[Dict[str, str]]

async def call_gemini_api(role: str, prompt: str, history: List[Dict]):
    await rate_limiter.wait_if_needed()
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction=ELYX_PERSONAS[role])
        chat_history = [{'role': 'user' if h['from'] == 'Rohan' else 'model', 'parts': [f"({h['from']}): {h['text']}"]} for h in history]
        
        # --- FIX: Enforce WhatsApp style in the final prompt ---
        final_prompt = f"({role}): {prompt} (Keep the response very brief, like a WhatsApp message. No more than 2 short sentences.)"

        response = await asyncio.to_thread(
            model.generate_content,
            [*chat_history, {'role': 'user', 'parts': [final_prompt]}],
            # --- FIX: Reduce max tokens to force shorter responses ---
            generation_config={"temperature": 0.7, "max_output_tokens": 30}
        )
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return f"[API Error: {str(e)}]"

@app.post("/start_simulation", response_model=List[SimulationResponse])
async def start_simulation(request: SimulationRequest):
    signup_date = datetime.fromisoformat("2025-01-01T09:00:00")
    events = generate_timeline(signup_date, months=request.months)
    
    full_simulation_output = []
    conversation_history = deque(maxlen=6)

    for event in events:
        event_conversation = []
        initiator = event['initiator']
        
        if event['event_type'] == 'member_message':
            details = event['metadata']['details']
            expert = event['metadata']['expert']
            prompt = f"As Rohan, start a conversation about: '{details}'."
            participants = [initiator, expert]
        elif event['event_type'] == 'travel_week':
            prompt = "As Rohan, mention you have a trip coming up and ask for advice on staying healthy."
            participants = [initiator, "Advik"]
        elif event['event_type'] == 'exercise_update':
            prompt = "As Rachel, proactively check in with Rohan about his biweekly exercise plan. Propose a small, specific adjustment."
            participants = [initiator, "Rohan"]
        elif event['event_type'] == 'diagnostic_panel':
            prompt = "As Ruby, remind Rohan about his upcoming diagnostic panel, confirming the time and any prep needed (e.g., fasting)."
            participants = [initiator, "Rohan"]
        else:
            continue

        current_prompt = prompt
        for role in participants:
            response_text = await call_gemini_api(role, current_prompt, list(conversation_history))
            turn = {"from": role, "text": response_text}
            event_conversation.append(turn)
            conversation_history.append(turn)
            current_prompt = "Based on the last message, write a concise, in-character response."

        full_simulation_output.append({"event": event, "conversation": event_conversation})

    return full_simulation_output

@app.get("/")
def read_root():
    return {"message": "Elyx Event-Driven Simulation Backend is running."}
