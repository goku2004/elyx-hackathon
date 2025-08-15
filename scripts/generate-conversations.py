# scripts/generate_conversations.py
import json, uuid, os
from datetime import datetime
from pathlib import Path
import random

PROMPT_DIR = Path("prompts")
RESP_DIR = PROMPT_DIR / "responses"
RESP_DIR.mkdir(parents=True, exist_ok=True)

# load role system prompts
with open(PROMPT_DIR / "role_systems.json") as f:
    ROLE_SYSTEMS = json.load(f)

# === Helper utilities ===
def make_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

def save_prompt_call(role, system_prompt, user_prompt, response_text, metadata=None):
    fn = RESP_DIR / f"{role}_{datetime.utcnow().isoformat().replace(':','-')}_{uuid.uuid4().hex[:6]}.json"
    payload = {
        "role": role,
        "system": system_prompt,
        "user": user_prompt,
        "response": response_text,
        "meta": metadata or {}
    }
    with open(fn, "w") as fh:
        json.dump(payload, fh, indent=2)
    return str(fn)

# === Config: allowed categories and weighting based on profile ===
DEFAULT_CATEGORIES = ["sleep", "travel logistics", "nutrition", "exercise", "symptom", "medication_query", "testing_query"]

def category_weights_for_profile(profile):
    # profile is a small dict with keys like 'travels_frequently', 'chronic_conditions'
    # For Rohan (travels frequently), give more weight to travel/sleep
    if profile.get("travels_frequently"):
        return {
            "sleep": 2.0,
            "travel logistics": 2.0,
            "nutrition": 1.0,
            "exercise": 1.0,
            "symptom": 1.0,
            "medication_query": 0.5,
            "testing_query": 0.5
        }
    # fallback uniform
    return {c: 1.0 for c in DEFAULT_CATEGORIES}

def weighted_choice(weight_map):
    items = list(weight_map.items())
    choices, weights = zip(*items)
    total = sum(weights)
    r = random.random() * total
    upto = 0
    for c, w in zip(choices, weights):
        if upto + w >= r:
            return c
        upto += w
    return choices[-1]

# === LLM client wrapper placeholder ===
def call_llm(system_prompt, user_prompt, temperature=0.3, max_tokens=250):
    """
    Replace this stub with your actual LLM client call.
    Example (OpenAI v1 chat): client.chat(messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], temperature=temperature)
    Return: text content string
    """
    # ====== FAKE STUB (for local testing) ======
    # Make the content plausibly reasonable for debugging if LLM not called.
    fake = f"[LLM simulated reply at temp={temperature}]"
    # In real run, remove above and call your SDK, returning response content.
    return fake

# === Generate member initial message (non-fixed template) ===
def generate_member_initial_message(profile, allowed_categories, role_system_prompt, seed=None):
    """
    Uses the member system prompt as system role and instructs the LLM to create a natural initial message.
    - profile: small dict describing member (strings)
    - allowed_categories: list of categories permitted for this event
    - role_system_prompt: system prompt for 'member' role loaded from JSON
    """
    # Build user prompt — *not* a rigid template for content, but constraints for content generation
    user_prompt = (
        "You are writing a single WhatsApp-style message as the member described below.\n\n"
        f"Member profile (short): {profile}\n\n"
        f"Allowed topics for this message (pick ONE and write a natural message): {allowed_categories}\n\n"
        "Constraints:\n"
        "- Produce a single natural message (1-2 sentences) that a busy 46-year-old frequent-traveler might send.\n"
        "- Use the persona tone: short, direct, sometimes mildly anxious.\n"
        "- If travel is selected, mention city or flight only if it's plausible (e.g., 'flight to London tomorrow').\n"
        "- Use one emoji at most, only if it feels natural.\n"
        "- Do NOT invent new chronic conditions. Keep content realistic and short.\n\n"
        "Do NOT return meta commentary. Return only the message text.\n"
    )

    # call LLM with member system prompt + user prompt
    system_prompt = role_system_prompt
    # temp higher for varied member voice
    response_text = call_llm(system_prompt, user_prompt, temperature=0.7, max_tokens=120)
    save_prompt_call("member_initial", system_prompt, user_prompt, response_text, metadata={"allowed": allowed_categories})
    return response_text.strip()

# === Example pipeline for an event ===
def process_event(event, profile, recent_context_msgs):
    """
    event: dict, e.g., {id, start_ts, event_type, metadata, initiator}
    profile: dict
    recent_context_msgs: list of last few messages for context (each dict with 'from','text')
    Returns: list of generated message dicts for that event
    """
    messages = []

    # load role system prompts
    member_sys = ROLE_SYSTEMS["member"]["system_prompt"]
    ruby_sys = ROLE_SYSTEMS["ruby"]["system_prompt"]
    dr_sys = ROLE_SYSTEMS["dr_warren"]["system_prompt"]
    carla_sys = ROLE_SYSTEMS["carla"]["system_prompt"]
    advik_sys = ROLE_SYSTEMS["advik"]["system_prompt"]

    if event["event_type"] == "member_message":
        # pick allowed categories (weighted by profile)
        weight_map = category_weights_for_profile(profile)
        chosen_cat = weighted_choice(weight_map)

        # craft initial natural member message via LLM
        initial_text = generate_member_initial_message(profile, [chosen_cat], member_sys)
        m1 = {"id": make_id("msg"), "event_id": event["id"], "ts": event["start_ts"], "from": "member", "to": "elyx", "text": initial_text, "topic": chosen_cat}
        messages.append(m1)

        # Elyx (Ruby) replies using context = last k messages + event metadata
        # Build a short user prompt for Ruby using context (keep this short to stay in tokens)
        ctx = "\n".join([f'{cm["from"]}: {cm["text"]}' for cm in (recent_context_msgs + [m1])][-3:])
        ruby_user_prompt = (
            f"Context:\n{ctx}\n\n"
            f"Event metadata: {event.get('metadata',{})}\n\n"
            "As Ruby, generate 1-2 short WhatsApp-style replies addressing the member query. If you need a clinician, propose triage and ask to schedule. Keep it short."
        )
        ruby_reply = call_llm(ruby_sys, ruby_user_prompt, temperature=0.25, max_tokens=200)
        save_prompt_call("ruby", ruby_sys, ruby_user_prompt, ruby_reply, metadata={"event": event["id"]})
        m2 = {"id": make_id("msg"), "event_id": event["id"], "ts": event["start_ts"], "from":"ruby", "to":"member", "text": ruby_reply}
        messages.append(m2)

        # Optionally generate a member follow-up (short ack)
        follow_user_ctx = "\n".join([f'{x["from"]}: {x["text"]}' for x in (recent_context_msgs + [m1, m2])][-3:])
        member_follow_prompt = (
            f"Context:\n{follow_user_ctx}\n\n"
            "As the member, write a single short reply (acknowledgement, quick question, or confirmation). Keep it 1 sentence."
        )
        member_follow = call_llm(member_sys, member_follow_prompt, temperature=0.6, max_tokens=80)
        save_prompt_call("member", member_sys, member_follow_prompt, member_follow, metadata={"event": event["id"], "followup": True})
        m3 = {"id": make_id("msg"), "event_id": event["id"], "ts": event["start_ts"], "from":"member", "to":"ruby", "text": member_follow}
        messages.append(m3)

    else:
        # e.g., exercise_update, diagnostic_panel, travel_week announcements by Ruby
        ctx = "\n".join([f'{cm["from"]}: {cm["text"]}' for cm in recent_context_msgs][-2:])
        ruby_user_prompt = f"Context:\n{ctx}\n\nEvent: {event['event_type']} | metadata: {event.get('metadata',{})}\n\nGenerate a short announcement message to the member."
        ruby_reply = call_llm(ruby_sys, ruby_user_prompt, temperature=0.2, max_tokens=200)
        save_prompt_call("ruby", ruby_sys, ruby_user_prompt, ruby_reply, metadata={"event": event["id"]})
        m = {"id": make_id("msg"), "event_id": event["id"], "ts": event["start_ts"], "from":"ruby", "to":"member", "text": ruby_reply}
        messages.append(m)

    return messages

# === Example run (pseudo) ===
if __name__ == "__main__":
    # Example inputs
    profile = {"name":"Rohan Patel", "travels_frequently": True, "chronic_conditions": ["hypertension"], "preferred_summary":"executive"}
    # load a few events (you would load data/events.json created earlier)
    example_event = {"id":"event-0001", "start_ts": datetime.utcnow().isoformat(), "event_type":"member_message", "initiator":"member", "metadata":{"topic_hint":""}}
    # recent_context_msgs could be loaded from data/messages.json (last 3 messages)
    recent_context_msgs = []
    msgs = process_event(example_event, profile, recent_context_msgs)
    print("Generated messages:", msgs)
