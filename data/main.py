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
from typing import List, Dict, Any, Optional
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

# --- Enhanced Event Generation Logic ---
MIN_EVENTS_PER_WEEK = 5
MAX_EVENTS_PER_WEEK = 8
TRAVEL_EVERY_N_WEEKS = 4
EXERCISE_UPDATE_EVERY_WEEKS = 2
DIAGNOSTIC_EVERY_MONTHS = 3
MEMBER_RESEARCH_CONVERSATIONS_PER_WEEK = 5
PLAN_ADHERENCE_RATE = 0.5
ROLE_MEMBER = "Rohan"

# Travel destinations (business trips from Singapore)
TRAVEL_DESTINATIONS = [
    "London", "Paris", "New York", "Tokyo", "Hong Kong", "Dubai", 
    "Sydney", "Mumbai", "New Delhi", "Bangkok", "Seoul", "Frankfurt",
    "Zurich", "San Francisco", "Shanghai", "Jakarta"
]

# Chronic conditions the member might be managing
CHRONIC_CONDITIONS = [
    {"condition": "pre_diabetes", "details": "Managing elevated HbA1c levels"},
    {"condition": "hypertension", "details": "Managing borderline high blood pressure"},
    {"condition": "high_cholesterol", "details": "Managing elevated LDL cholesterol"},
    {"condition": "sleep_apnea", "details": "Managing mild sleep apnea with lifestyle changes"},
    {"condition": "vitamin_d_deficiency", "details": "Managing low vitamin D levels"}
]

# Expanded member topics for research conversations
MEMBER_TOPICS = [
    {"topic": "garmin_hrv_low", "details": "Low HRV reading on Garmin watch", "expert": "Advik"},
    {"topic": "poor_sleep_score", "details": "Concern about a low sleep score", "expert": "Advik"},
    {"topic": "nutrition_question", "details": "Question about a specific food or diet plan", "expert": "Carla"},
    {"topic": "exercise_soreness", "details": "Unusual muscle soreness after a workout", "expert": "Rachel"},
    {"topic": "symptom_check", "details": "Minor symptom check, e.g., headache or fatigue", "expert": "Dr. Warren"},
    {"topic": "curiosity_longevity", "details": "A general question about a health or longevity topic", "expert": "Neel"},
    {"topic": "supplement_research", "details": "Research question about a supplement or biohack", "expert": "Dr. Warren"},
    {"topic": "workout_modification", "details": "Request to modify workout due to schedule constraints", "expert": "Rachel"},
    {"topic": "meal_prep_help", "details": "Need help with meal prep for busy week", "expert": "Carla"},
    {"topic": "travel_nutrition", "details": "How to maintain nutrition while traveling", "expert": "Carla"},
    {"topic": "stress_management", "details": "Question about managing work stress and its health impact", "expert": "Neel"},
    {"topic": "biometric_interpretation", "details": "Help interpreting recent health metrics or wearable data", "expert": "Advik"},
    {"topic": "recovery_optimization", "details": "How to optimize recovery between workouts", "expert": "Rachel"},
    {"topic": "cognitive_enhancement", "details": "Question about nootropics or cognitive performance", "expert": "Dr. Warren"},
    {"topic": "lab_result_question", "details": "Question about interpreting recent lab results", "expert": "Dr. Warren"}
]

# Plan adherence tracking topics
PLAN_ADHERENCE_TOPICS = [
    {"topic": "diet_compliance_issue", "details": "Struggling to follow prescribed diet plan", "requires_adjustment": True},
    {"topic": "exercise_schedule_conflict", "details": "Can't fit exercise into current schedule", "requires_adjustment": True},
    {"topic": "supplement_routine_problem", "details": "Difficulty maintaining supplement routine", "requires_adjustment": True},
    {"topic": "plan_working_well", "details": "Current plan is working great, feeling good", "requires_adjustment": False},
    {"topic": "partial_compliance", "details": "Following plan mostly but need minor adjustments", "requires_adjustment": True}
]

def make_id(prefix): 
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

def iso(dt): 
    return dt.isoformat()

def generate_poisson_events(rate_per_week, weeks):
    """Generate events using Poisson distribution"""
    events_per_week = np.random.poisson(rate_per_week, weeks)
    return events_per_week

def select_chronic_condition():
    """Select one chronic condition for the member to manage"""
    return np.random.choice(CHRONIC_CONDITIONS)

def generate_timeline(signup_date: datetime, months=1):
    timeline = []
    total_weeks = int(months * 4.345)
    cur = signup_date
    
    # Select one chronic condition for this member
    member_condition = select_chronic_condition()
    
    # Generate diagnostic panel dates (every 3 months)
    diag_dates = [signup_date + relativedelta(months=m) for m in range(0, months + 1, DIAGNOSTIC_EVERY_MONTHS)]
    
    # Use Poisson distribution for member research conversations
    research_conversations_per_week = generate_poisson_events(MEMBER_RESEARCH_CONVERSATIONS_PER_WEEK, total_weeks)

    for week_index in range(total_weeks):
        week_start = cur
        week_end = cur + timedelta(days=7)
        weekly_events = []
        
        # 1. TRAVEL EVENTS (1 week out of every 4 weeks)
        if (week_index + 1) % TRAVEL_EVERY_N_WEEKS == 0:
            destination = np.random.choice(TRAVEL_DESTINATIONS)
            weekly_events.append({
                "id": make_id("event"), "start_ts": iso(week_start), "event_type": "travel_week", 
                "initiator": ROLE_MEMBER, "metadata": {
                    "destination": destination, "note": f"Business trip to {destination}",
                    "duration_days": np.random.randint(3, 8)
                }
            })

        # 2. EXERCISE UPDATES (every 2 weeks)
        if (week_index + 1) % EXERCISE_UPDATE_EVERY_WEEKS == 0 and week_index > 0:
            ts = week_start + timedelta(days=np.random.randint(1, 3), hours=9)
            needs_adjustment = np.random.random() < (1 - PLAN_ADHERENCE_RATE)
            adherence_topic = np.random.choice(PLAN_ADHERENCE_TOPICS)
            
            weekly_events.append({
                "id": make_id("event"), "start_ts": iso(ts), "event_type": "exercise_update", 
                "initiator": "Rachel", "metadata": {
                    "details": "Biweekly exercise plan update", "needs_adjustment": needs_adjustment,
                    "adherence_feedback": adherence_topic["details"] if needs_adjustment else "Plan adherence is good"
                }
            })

        # 3. DIAGNOSTIC PANELS (every 3 months)
        for d in diag_dates:
            if week_start <= d < week_end:
                weekly_events.append({
                    "id": make_id("event"), "start_ts": iso(d + timedelta(hours=9)), 
                    "event_type": "diagnostic_panel", "initiator": "Ruby", "metadata": {
                        "panel": "Comprehensive health panel",
                        "tests": ["Complete Blood Count", "Lipid Profile", "HbA1c", "Vitamin D", "Inflammatory Markers"],
                        "chronic_condition_focus": member_condition["condition"]
                    }
                })

        # 4. CHRONIC CONDITION MANAGEMENT (30% chance per week)
        if week_index > 2 and np.random.random() < 0.3:
            ts = week_start + timedelta(days=np.random.randint(0, 6), hours=np.random.randint(8, 18))
            weekly_events.append({
                "id": make_id("event"), "start_ts": iso(ts), "event_type": "chronic_condition_check", 
                "initiator": ROLE_MEMBER, "metadata": {
                    "condition": member_condition["condition"],
                    "details": f"Check-in regarding {member_condition['details']}",
                    "expert": "Dr. Warren"
                }
            })

        # 5. PLAN ADHERENCE CHECK (40% chance per week)
        if week_index > 0 and np.random.random() < 0.4:
            ts = week_start + timedelta(days=np.random.randint(0, 6), hours=np.random.randint(9, 17))
            adherence_topic = np.random.choice(PLAN_ADHERENCE_TOPICS)
            weekly_events.append({
                "id": make_id("event"), "start_ts": iso(ts), "event_type": "plan_adherence_check", 
                "initiator": ROLE_MEMBER, "metadata": {
                    "adherence_rate": PLAN_ADHERENCE_RATE, "topic": adherence_topic["topic"],
                    "details": adherence_topic["details"], 
                    "requires_plan_adjustment": adherence_topic["requires_adjustment"],
                    "expert": "Neel"
                }
            })

        # 6. PROGRESS REPORTS (monthly)
        if week_index > 0 and week_index % 4 == 0:
            ts = week_start + timedelta(days=1, hours=10)
            weekly_events.append({
                "id": make_id("event"), "start_ts": iso(ts), "event_type": "progress_report", 
                "initiator": "Dr. Warren", "metadata": {
                    "report_type": "Monthly progress summary",
                    "focus_areas": ["Biomarker trends", "Exercise compliance", "Nutrition adherence"],
                    "chronic_condition_status": member_condition["condition"]
                }
            })

        # 7. MEMBER RESEARCH CONVERSATIONS (using Poisson distribution, 5-7 per week)
        num_research_conversations = min(research_conversations_per_week[week_index], 7)
        
        for _ in range(num_research_conversations):
            ts = week_start + timedelta(days=np.random.randint(0, 6), hours=np.random.randint(7, 22))
            topic_index = np.random.randint(0, len(MEMBER_TOPICS))
            chosen_topic = MEMBER_TOPICS[topic_index]
            weekly_events.append({
                "id": make_id("event"), "start_ts": iso(ts), "event_type": "member_research_question", 
                "initiator": ROLE_MEMBER, "metadata": {
                    **chosen_topic, "conversation_type": "research_curiosity", "time_commitment": "5-10 minutes"
                }
            })

        # 8. ELYX PROACTIVE CHECK-INS (60% chance per week)
        if np.random.random() < 0.6:
            ts = week_start + timedelta(days=np.random.randint(1, 4), hours=np.random.randint(9, 11))
            elyx_members = ["Ruby", "Neel", "Dr. Warren", "Advik", "Carla", "Rachel"]
            initiator = np.random.choice(elyx_members)
            weekly_events.append({
                "id": make_id("event"), "start_ts": iso(ts), "event_type": "elyx_proactive_checkin", 
                "initiator": initiator, "metadata": {
                    "check_type": "Proactive support",
                    "details": f"{initiator} checking in on member progress and needs"
                }
            })

        timeline.extend(weekly_events)
        cur = week_end

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

# Add the missing GenerateRequest model for the /generate endpoint
class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    role: Optional[str] = "Rohan"
    context: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = []

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
        event_type = event['event_type']
        
        # Handle different event types with appropriate prompts and participants
        if event_type == 'member_research_question':
            details = event['metadata']['details']
            expert = event['metadata']['expert']
            prompt = f"As Rohan, start a conversation about: '{details}'."
            participants = [initiator, expert]
            
        elif event_type == 'travel_week':
            destination = event['metadata'].get('destination', 'London')
            prompt = f"As Rohan, mention you have a business trip to {destination} coming up and ask for advice on staying healthy."
            participants = [initiator, "Advik"]
            
        elif event_type == 'exercise_update':
            needs_adjustment = event['metadata'].get('needs_adjustment', False)
            if needs_adjustment:
                prompt = "As Rachel, check in with Rohan about his exercise plan. Mention you've noticed some adherence issues and propose adjustments."
            else:
                prompt = "As Rachel, proactively check in with Rohan about his biweekly exercise plan. Propose a small, specific progression."
            participants = [initiator, "Rohan"]
            
        elif event_type == 'diagnostic_panel':
            panel_type = event['metadata'].get('panel', 'Full blood panel')
            prompt = f"As Ruby, remind Rohan about his upcoming {panel_type}, confirming the time and any prep needed (e.g., fasting)."
            participants = [initiator, "Rohan"]
            
        elif event_type == 'chronic_condition_check':
            condition = event['metadata']['condition']
            details = event['metadata']['details']
            expert = event['metadata']['expert']
            prompt = f"As Rohan, check in about your {condition}. Mention: '{details}'."
            participants = [initiator, expert]
            
        elif event_type == 'plan_adherence_check':
            details = event['metadata']['details']
            requires_adjustment = event['metadata'].get('requires_plan_adjustment', False)
            expert = event['metadata']['expert']
            if requires_adjustment:
                prompt = f"As Rohan, mention you're having issues: '{details}'. Ask for help adjusting the plan."
            else:
                prompt = f"As Rohan, give positive feedback: '{details}'. Ask if you should continue."
            participants = [initiator, expert]
            
        elif event_type == 'progress_report':
            focus_areas = event['metadata'].get('focus_areas', ['General health'])
            areas_str = ', '.join(focus_areas)
            prompt = f"As Dr. Warren, provide a brief progress update focusing on: {areas_str}. Highlight key trends."
            participants = [initiator, "Rohan"]
            
        elif event_type == 'elyx_proactive_checkin':
            check_type = event['metadata'].get('check_type', 'General check-in')
            prompt = f"As {initiator}, do a proactive {check_type.lower()}. Ask how Rohan is doing and if he needs support."
            participants = [initiator, "Rohan"]
            
        elif event_type == 'member_message':  # Legacy support
            details = event['metadata']['details']
            expert = event['metadata']['expert']
            prompt = f"As Rohan, start a conversation about: '{details}'."
            participants = [initiator, expert]
            
        else:
            continue  # Skip unknown event types

        current_prompt = prompt
        for role in participants:
            response_text = await call_gemini_api(role, current_prompt, list(conversation_history))
            turn = {"from": role, "text": response_text}
            event_conversation.append(turn)
            conversation_history.append(turn)
            current_prompt = "Based on the last message, write a concise, in-character response."

        full_simulation_output.append({"event": event, "conversation": event_conversation})

    return full_simulation_output

# Add the missing /generate endpoint
@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        if not request.prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        role = request.role or "Rohan"
        history = request.history or []
        
        # Validate role exists
        if role not in ELYX_PERSONAS:
            raise HTTPException(status_code=400, detail=f"Invalid role: {role}. Available roles: {list(ELYX_PERSONAS.keys())}")
        
        response_text = await call_gemini_api(role, request.prompt, history)
        
        return {
            "text": response_text,  # Frontend expects "text" field
            "response": response_text,  # Keep this for compatibility
            "role": role,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Generate endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Elyx Event-Driven Simulation Backend is running."}