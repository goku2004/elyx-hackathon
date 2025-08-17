# scripts/generate_events.py
import json
import uuid
from datetime import datetime, timedelta
import numpy as np
from dateutil.relativedelta import relativedelta
from pathlib import Path

# CONFIG
DATA_DIR = Path(__file__).resolve().parent.parent/"data"/"events.json"
DATA_DIR.parent.mkdir(exist_ok=True, parents=True)

SEED = 42
np.random.seed(SEED)

DEFAULT_MONTHS = 8
MIN_EVENTS_PER_WEEK = 4
MAX_EVENTS_PER_WEEK = 7

# rules
TRAVEL_EVERY_N_WEEKS = 4
EXERCISE_UPDATE_EVERY_WEEKS = 2
DIAGNOSTIC_EVERY_MONTHS = 3

ROLE_MEMBER = "member"
ROLE_ELYX = "elyx_team" 

# More specific topics for member-initiated events
MEMBER_TOPICS = [
    {"topic": "garmin_hrv_low", "details": "Low HRV reading on Garmin watch"},
    {"topic": "poor_sleep_score", "details": "Concern about a low sleep score"},
    {"topic": "nutrition_question", "details": "Question about a specific food or diet plan"},
    {"topic": "exercise_soreness", "details": "Unusual muscle soreness after a workout"},
    {"topic": "symptom_check", "details": "Minor symptom check, e.g., headache or fatigue"},
    {"topic": "curiosity_longevity", "details": "A general question about a health or longevity topic"}
]

def iso(dt):
    return dt.isoformat()

def make_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

def generate_timeline(signup_date: datetime, months=DEFAULT_MONTHS):
    timeline = []
    total_weeks = int(months * 4.345) # Use a more precise week calculation
    cur = signup_date
    
    diag_dates = [signup_date + relativedelta(months=m) for m in range(0, months + 1, DIAGNOSTIC_EVERY_MONTHS)]

    for week_index in range(total_weeks):
        week_start = cur
        week_end = cur + timedelta(days=7)
        
        weekly_events = []
        
        # Add fixed, scheduled events for the week
        # Use week_index + 1 for more natural 1-based counting
        if (week_index + 1) % TRAVEL_EVERY_N_WEEKS == 0:
            weekly_events.append({
                "id": make_id("event"), "start_ts": iso(week_start), "end_ts": iso(week_end),
                "event_type": "travel_week", "initiator": ROLE_MEMBER,
                "metadata": {"note": "Member is traveling this week"}
            })

        if (week_index + 1) % EXERCISE_UPDATE_EVERY_WEEKS == 0 and week_index > 0:
            ts = week_start + timedelta(days=np.random.randint(1,3), hours=9)
            weekly_events.append({
                "id": make_id("event"), "start_ts": iso(ts),
                "event_type": "exercise_update", "initiator": ROLE_ELYX,
                "metadata": {"details": "Biweekly exercise plan update"}
            })

        for d in diag_dates:
            if week_start <= d < week_end:
                weekly_events.append({
                    "id": make_id("event"), "start_ts": iso(d + timedelta(hours=9)),
                    "event_type": "diagnostic_panel", "initiator": ROLE_ELYX,
                    "metadata": {"panel": "Full blood panel"}
                })
        
        # Calculate how many member-initiated events to add
        target_events_for_week = np.random.randint(MIN_EVENTS_PER_WEEK, MAX_EVENTS_PER_WEEK + 1)
        member_events_to_add = max(0, target_events_for_week - len(weekly_events))

        # Add member-initiated events
        for _ in range(member_events_to_add):
            delta = timedelta(days=np.random.randint(0, 6), hours=np.random.randint(7, 22))
            ts = week_start + delta
            # Use direct indexing for numpy.random.choice with a list of dicts
            topic_index = np.random.randint(0, len(MEMBER_TOPICS))
            chosen_topic = MEMBER_TOPICS[topic_index]
            weekly_events.append({
                "id": make_id("event"), "start_ts": iso(ts),
                "event_type": "member_message", "initiator": ROLE_MEMBER,
                "metadata": chosen_topic
            })
            
        timeline.extend(weekly_events)
        cur = week_end

    timeline.sort(key=lambda e: e['start_ts'])
    return timeline

if __name__ == "__main__":
    signup = datetime.fromisoformat("2025-01-01T09:00:00")
    events = generate_timeline(signup, months=8)
    with open(DATA_DIR, "w") as f:
        json.dump(events, f, indent=2)
    print(f"Generated {len(events)} events -> {DATA_DIR}")
