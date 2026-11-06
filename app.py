"""Context-aware chatbot backed by the Gemini API.
 
Context strategy (hybrid):
  1. Keep the full message history per conversation.
  2. Once it exceeds MAX_HISTORY_MESSAGES, the oldest messages are summarised by
     the LLM into a running summary, which is injected into the system prompt.
     Only the most recent KEEP_RECENT_MESSAGES stay verbatim.
  3. If the summarisation call fails, we fall back to plain truncation so the
     chat never breaks.
"""
import logging
import os
import threading
 
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import errors, types
 
load_dotenv()
 
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
 
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_HISTORY = int(os.getenv("MAX_HISTORY_MESSAGES", 12))   # trigger point
KEEP_RECENT = int(os.getenv("KEEP_RECENT_MESSAGES", 6))    # kept verbatim (even number!)
MAX_INPUT_CHARS = 4000
TIMEOUT_MS = 30_000
 
SYSTEM_PROMPT = """You are Sage, a friendly and patient study coach for beginner programmers.
- Explain concepts with short, concrete examples; prefer analogies over jargon.
- Ask one clarifying question when the user's goal is unclear.
- Keep answers under ~150 words unless the user asks for more depth.
- If asked about something unrelated to learning, programming, or tech careers,
  politely steer back to what you can help with.
- Never invent facts; say so when you are unsure.
- Remember what the user told you earlier (their name, goals, language) and use it."""
 
SUMMARY_PROMPT = """Update the running summary of a conversation between a user and a study coach.
Keep every durable fact: the user's name, goals, skill level, preferences, topics covered,
and any open questions. Be concise (max 120 words). Output only the summary text."""
 
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("chatbot")
 
client = genai.Client(api_key=API_KEY, http_options=types.HttpOptions(timeout=TIMEOUT_MS))
app = Flask(__name__, static_folder="static")
 
# conversation_id -> {"history": [{"role": "user"|"model", "text": str}], "summary": str}
conversations: dict[str, dict] = {}
lock = threading.Lock()
 
 
def to_contents(history):
    return [types.Content(role=m["role"], parts=[types.Part(text=m["text"])]) for m in history]
 
 
def summarise(old_summary: str, old_messages: list[dict]) -> str:
    transcript = "\n".join(f"{m['role']}: {m['text']}" for m in old_messages)
    prompt = f"Previous summary:\n{old_summary or '(none)'}\n\nNew messages to fold in:\n{transcript}"
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SUMMARY_PROMPT, temperature=0.2),
    )
    return (resp.text or "").strip()
 
 
def manage_context(convo: dict) -> None:
    """Compress or truncate history once it grows past MAX_HISTORY."""
    history = convo["history"]
    if len(history) <= MAX_HISTORY:
        return
    cut = len(history) - KEEP_RECENT
    cut -= cut % 2  # cut on a user/model pair boundary so history still starts with "user"
    old, recent = history[:cut], history[cut:]
    try:
        convo["summary"] = summarise(convo["summary"], old)
        log.info("Summarised %d old messages", len(old))
    except Exception as exc:  # fallback: truncate (older info is dropped)
        log.warning("Summarisation failed (%s); falling back to truncation", exc)
    convo["history"] = recent
 
 
def friendly_error(exc: Exception) -> tuple[str, int]:
    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        if code == 429:
            return "I'm being rate-limited right now. Please wait a few seconds and try again.", 429
        if code in (401, 403):
            return "The server's API key is invalid or lacks permission. Please contact the admin.", 502
        if code == 400:
            return "The AI service rejected that request. Try rephrasing your message.", 400
        if code and code >= 500:
            return "The AI service is having trouble. Please try again shortly.", 503
    text = str(exc).lower()
    if "timed out" in text or "timeout" in text:
        return "The AI took too long to respond. Please try again.", 504
    return "Something went wrong talking to the AI. Please try again.", 500
 
 
@app.get("/")
def index():
    return send_from_directory("static", "index.html")
 
 
@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    cid = str(data.get("conversation_id", "")).strip()
    message = str(data.get("message", "")).strip()
    if not cid or not message:
        return jsonify(error="conversation_id and message are required."), 400
    if len(message) > MAX_INPUT_CHARS:
        return jsonify(error=f"Message too long (max {MAX_INPUT_CHARS} characters)."), 400
 
    with lock:
        convo = conversations.setdefault(cid, {"history": [], "summary": ""})
        manage_context(convo)
        system = SYSTEM_PROMPT
        if convo["summary"]:
            system += f"\n\nSummary of the earlier conversation (for your memory):\n{convo['summary']}"
        contents = to_contents(convo["history"] + [{"role": "user", "text": message}])
 
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system, temperature=0.7),
            )
            reply = (resp.text or "").strip()
            if not reply:
                return jsonify(error="The AI returned an empty response (it may have been blocked). Try rephrasing."), 502
        except Exception as exc:
            log.exception("LLM call failed")
            msg, status = friendly_error(exc)
            return jsonify(error=msg), status  # history untouched -> user can retry
 
        # Only commit the turn after success so roles keep alternating.
        convo["history"] += [{"role": "user", "text": message}, {"role": "model", "text": reply}]
        return jsonify(reply=reply, history_length=len(convo["history"]), has_summary=bool(convo["summary"]))
 
 
@app.post("/api/reset")
def reset():
    cid = str((request.get_json(silent=True) or {}).get("conversation_id", ""))
    with lock:
        conversations.pop(cid, None)
    return jsonify(ok=True)
 
 
if __name__ == "__main__":
    app.run(debug=False, port=int(os.getenv("PORT", 5000)))
 

