# scripts/generate_events.py
import json
import uuid
from datetime import datetime, timedelta
import numpy as np
from dateutil.relativedelta import relativedelta
from pathlib import Path

# CONFIG
DATA_DIR = Path(__file__).resolve().parent.parent/"data"/"events.json"

SEED = 42
np.random.seed(SEED)

WEEKS_PER_MONTH = 4.345  # approximate
DEFAULT_MONTHS = 8

# rules
MAX_MEMBER_CONV_PER_WEEK = 5
TRAVEL_EVERY_N_WEEKS = 4
EXERCISE_UPDATE_EVERY_WEEKS = 2
DIAGNOSTIC_EVERY_MONTHS = 3

ROLE_MEMBER = "member"
ROLE_RUBY = "ruby"  # concierge
# add other roles if needed

def iso(dt):
    return dt.isoformat()

def make_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

def generate_timeline(signup_date: datetime, months=DEFAULT_MONTHS):
    timeline = []
    total_days = int(months * 30.4375)
    end_date = signup_date + timedelta(days=total_days)
    cur = signup_date
    week_index = 0

    # compute diagnostic months (0,3,6,...)
    diag_dates = [signup_date + relativedelta(months=m) for m in range(0, months+1, DIAGNOSTIC_EVERY_MONTHS)]

    while cur < end_date:
        week_start = cur
        week_end = cur + timedelta(days=7)
        # travel week if it's the Nth week
        if (week_index % TRAVEL_EVERY_N_WEEKS) == 0:
            timeline.append({
                "id": make_id("event"),
                "start_ts": iso(week_start),
                "end_ts": iso(week_end),
                "event_type": "travel_week",
                "initiator": ROLE_MEMBER,
                "metadata": {"note": "member traveling; timezone shift likely"}
            })

        # member-initiated conversation count (Poisson with mean ~2.5 but cap)
        member_convs = min(MAX_MEMBER_CONV_PER_WEEK, int(np.random.poisson(2.2)))
        for i in range(member_convs):
            # space them in the week
            delta = timedelta(days=np.random.randint(0,7), hours=np.random.randint(7,22))
            ts = week_start + delta
            timeline.append({
                "id": make_id("event"),
                "start_ts": iso(ts),
                "event_type": "member_message",
                "initiator": ROLE_MEMBER,
                "metadata": {"topic": np.random.choice(["sleep", "travel logistics", "nutrition", "exercise", "symptom"])}
            })

        # periodic exercise update every 2 weeks
        if (week_index % EXERCISE_UPDATE_EVERY_WEEKS) == 0:
            ts = week_start + timedelta(days=1, hours=9)
            timeline.append({
                "id": make_id("event"),
                "start_ts": iso(ts),
                "event_type": "exercise_update",
                "initiator": ROLE_RUBY,
                "metadata": {"details": "biweekly exercise plan update"}
            })

        # scheduled diagnostics
        for d in diag_dates:
            # check if d is within this week
            if week_start <= d < week_end:
                timeline.append({
                    "id": make_id("event"),
                    "start_ts": iso(d + timedelta(hours=9)),
                    "event_type": "diagnostic_panel",
                    "initiator": "lab_system",
                    "metadata": {"panel": "full_blood_panel"}
                })

        cur = week_end
        week_index += 1

    # Sort by start_ts and return
    timeline.sort(key=lambda e: e['start_ts'])
    return timeline

if __name__ == "__main__":
    signup = datetime.fromisoformat("2025-01-01T09:00:00")
    events = generate_timeline(signup, months=8)
    with open(DATA_DIR, "w") as f:
        json.dump(events, f, indent=2)
    print(f"Generated {len(events)} events -> data/events.json")
