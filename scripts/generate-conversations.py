# scripts/generate_conversations.py
import os
import json
import uuid
import random
from datetime import datetime
from pathlib import Path
from dateutil import parser as dateparser

# --- OpenAI client ---
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("Install the openai package: pip install openai") from e

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY environment variable before running.")

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Paths (script-relative -> repo root) ---
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
DATA_DIR = REPO_ROOT / "data"
PROMPT_DIR = REPO_ROOT / "prompts"
RESP_DIR = PROMPT_DIR / "responses"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESP_DIR.mkdir(parents=True, exist_ok=True)

ROLE_SYSTEMS_PATH = PROMPT_DIR / "role_systems.json"
if not ROLE_SYSTEMS_PATH.exists():
    raise FileNotFoundError(f"Missing role_systems.json at {ROLE_SYSTEMS_PATH}")

# --- Load role system prompts ---
with open(ROLE_SYSTEMS_PATH, "r", encoding="utf-8") as fh:
    ROLE_SYSTEMS = json.load(fh)

# --- Utilities ---
def make_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

def now_iso():
    return datetime.utcnow().isoformat()

def save_prompt_call(role_key, system_prompt, user_prompt, raw_response_obj, meta=None):
    """Save system+user prompt and relevant response text + some raw info for provenance."""
    ts = datetime.utcnow().isoformat().replace(":", "-")
    filename = RESP_DIR / f"{role_key}_{ts}_{uuid.uuid4().hex[:6]}.json"

    # Try to get representative text from the response object
    response_text = extract_response_text(raw_response_obj)

    payload = {
        "role_key": role_key,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response_text": response_text,
        "meta": meta or {},
        # store a minimal serialized raw representation (string) to help debugging
        "raw_response_str": safe_str(raw_response_obj),
        "saved_at": now_iso()
    }
    with open(filename, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return str(filename)

def safe_str(obj):
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"

def extract_response_text(resp):
    """
    Try a few common shapes to extract assistant text from an OpenAI response object.
    """
    if resp is None:
        return ""
    # 1: newer SDK: resp.output_text
    text = getattr(resp, "output_text", None)
    if text:
        return text.strip()
    # 2: sometimes resp is dict-like
    try:
        if isinstance(resp, dict):
            # common shape: resp['output'][0]['content'][0]['text']
            out = resp.get("output")
            if out and isinstance(out, list):
                parts = []
                for item in out:
                    if isinstance(item, dict) and "content" in item:
                        for c in item["content"]:
                            if isinstance(c, dict) and c.get("type") == "output_text":
                                parts.append(c.get("text",""))
                            elif isinstance(c, str):
                                parts.append(c)
                if parts:
                    return " ".join(parts).strip()
            # fallback to completion or text keys
            for k in ("text", "completion", "response"):
                if k in resp:
                    return str(resp[k]).strip()
    except Exception:
        pass
    # 3: fallback to str()
    return safe_str(resp)

# --- OpenAI call wrapper ---
def call_openai_with_prompts(system_prompt: str, user_prompt: str, model="gpt-4o-mini", temperature=0.3, max_tokens=300):
    """
    Calls OpenAI Responses API and returns the response object (and we extract text for saving).
    """
    combined_input = f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"
    try:
        response = client.responses.create(
            model=model,
            input=combined_input,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
    except Exception as e:
        # Provide helpful debugging message; re-raise to fail loudly in CI/hackathon
        raise RuntimeError(f"OpenAI call failed: {e}") from e

    return response

def call_llm(role_key, system_prompt, user_prompt, model="gpt-4o-mini", temperature=0.3, max_tokens=300, meta=None):
    """
    Unified wrapper used by the generator. Returns the assistant text (string).
    Also saves prompt+response to prompts/responses/.
    """
    resp = call_openai_with_prompts(system_prompt, user_prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    save_prompt_call(role_key, system_prompt, user_prompt, resp, meta=meta)
    text = extract_response_text(resp)
    return text

# --- Simple category weighting helper (persona-specific) ---
DEFAULT_CATEGORIES = ["sleep", "travel logistics", "nutrition", "exercise", "symptom", "medication_query", "testing_query"]
def category_weights_for_profile(profile):
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

# --- Message generation functions ---
def generate_member_initial_message(profile, allowed_categories, member_sys_prompt):
    """Ask the LLM to write a natural member-initiated message constrained to allowed_categories."""
    user_prompt = (
        "You will write a single WhatsApp-style message as the member described below.\n\n"
        f"Member profile (short): {profile}\n\n"
        f"Allowed topics (pick ONE): {allowed_categories}\n\n"
        "Constraints:\n"
        "- Produce a single natural message (1-2 sentences) that a busy 46-year-old frequent-traveler might send.\n"
        "- Use the persona tone: short, direct, sometimes mildly anxious.\n"
        "- If travel is selected, mention city or flight only if it's plausible.\n"
        "- Use at most one emoji and only if it feels natural.\n"
        "- Do NOT invent new chronic conditions.\n\n"
        "Return only the message text (no explanations)."
    )
    # member messages should be a bit more creative
    text = call_llm("member_initial", member_sys_prompt, user_prompt, model="gpt-4o-mini", temperature=0.7, max_tokens=120, meta={"kind":"member_initial", "allowed": allowed_categories})
    return text.strip()

def process_event(event, profile, recent_context_msgs):
    """
    event: dict with keys {id, start_ts, event_type, metadata, initiator}
    recent_context_msgs: list of {from, text} latest messages for context (optional)
    returns: list of message dicts generated for this event
    """
    messages = []

    # pull system prompts
    member_sys = ROLE_SYSTEMS["member"]["system_prompt"]
    ruby_sys = ROLE_SYSTEMS["ruby"]["system_prompt"]
    dr_sys = ROLE_SYSTEMS["dr_warren"]["system_prompt"]
    carla_sys = ROLE_SYSTEMS["carla"]["system_prompt"]
    advik_sys = ROLE_SYSTEMS["advik"]["system_prompt"]

    if event["event_type"] == "member_message":
        # pick allowed category using profile weighting
        weight_map = category_weights_for_profile(profile)
        chosen_cat = weighted_choice(weight_map)

        # generate natural initial message
        initial_text = generate_member_initial_message(profile, [chosen_cat], member_sys)
        m1 = {"id": make_id("msg"), "event_id": event["id"], "ts": event["start_ts"], "from": "member", "to": "ruby", "text": initial_text, "topic": chosen_cat}
        messages.append(m1)

        # Create short context string (last 3 messages)
        ctx_msgs = (recent_context_msgs + [m1])[-3:]
        ctx_str = "\n".join([f'{m["from"]}: {m["text"]}' for m in ctx_msgs])

        # Ruby reply
        ruby_user_prompt = (
            f"Context:\n{ctx_str}\n\n"
            f"Event metadata: {event.get('metadata',{})}\n\n"
            "As Ruby, generate 1-2 short WhatsApp-style replies addressing the member query. "
            "If scheduling is needed propose 2 timezone-aware time options. If clinical input is needed, triage and propose clinician consult."
        )
        ruby_reply = call_llm("ruby", ruby_sys, ruby_user_prompt, model="gpt-4o-mini", temperature=0.25, max_tokens=200, meta={"event": event["id"]})
        m2 = {"id": make_id("msg"), "event_id": event["id"], "ts": event["start_ts"], "from":"ruby", "to":"member", "text": ruby_reply}
        messages.append(m2)

        # Member follow-up ack (one short sentence)
        follow_ctx = "\n".join([f'{x["from"]}: {x["text"]}' for x in (recent_context_msgs + [m1, m2])][-3:])
        member_follow_prompt = f"Context:\n{follow_ctx}\n\nAs the member, write a single short reply (acknowledgement or quick question)."
        member_follow = call_llm("member", member_sys, member_follow_prompt, model="gpt-4o-mini", temperature=0.6, max_tokens=80, meta={"event": event["id"], "followup": True})
        m3 = {"id": make_id("msg"), "event_id": event["id"], "ts": event["start_ts"], "from":"member", "to":"ruby", "text": member_follow}
        messages.append(m3)

    else:
        # Non-member event: Ruby posts announcement
        ctx_str = "\n".join([f'{m["from"]}: {m["text"]}' for m in recent_context_msgs][-2:])
        ruby_user_prompt = f"Context:\n{ctx_str}\n\nEvent: {event['event_type']} | metadata: {event.get('metadata',{})}\n\nGenerate a short announcement message to the member."
        ruby_reply = call_llm("ruby", ruby_sys, ruby_user_prompt, model="gpt-4o-mini", temperature=0.2, max_tokens=180, meta={"event": event["id"]})
        m = {"id": make_id("msg"), "event_id": event["id"], "ts": event["start_ts"], "from":"ruby", "to":"member", "text": ruby_reply}
        messages.append(m)

    return messages

# --- Main runner: iterate events and build messages.json ---
def main():
    events_file = DATA_DIR / "events.json"
    out_messages_file = DATA_DIR / "messages.json"

    if not events_file.exists():
        raise FileNotFoundError(f"{events_file} not found. Run your event generator first.")

    with open(events_file, "r", encoding="utf-8") as fh:
        events = json.load(fh)

    profile = {"name":"Rohan Patel", "travels_frequently": True, "chronic_conditions": ["hypertension"], "preferred_summary":"executive"}

    # For simplicity, we don't maintain a full chronological context across all events in this script.
    # You can expand by tracking messages list and passing last N messages as recent_context_msgs to process_event.
    all_messages = []
    recent = []  # keep last few messages for context passing

    for ev in events:
        # ensure start_ts is ISO string (if date objects exist, convert)
        if isinstance(ev.get("start_ts"), str):
            # ok
            pass
        else:
            ev["start_ts"] = datetime.utcnow().isoformat()

        msgs = process_event(ev, profile, recent_context_msgs=recent[-5:])
        all_messages.extend(msgs)
        # append messages to recent for next iterations
        recent.extend(msgs)
        # keep recent bounded
        if len(recent) > 50:
            recent = recent[-50:]

    # dump to data/messages.json
    with open(out_messages_file, "w", encoding="utf-8") as fh:
        json.dump(all_messages, fh, ensure_ascii=False, indent=2)

    print(f"Wrote {len(all_messages)} messages to {out_messages_file}")
    print(f"Saved LLM prompt/response artifacts to {RESP_DIR}")

if __name__ == "__main__":
    main()
