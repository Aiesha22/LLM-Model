
# Sage – Context-Aware Chatbot (Flask + Gemini)
 
A multi-turn study-coach chatbot with real context-window management.
 
## Features
- `/api/chat` endpoint → Gemini API (`google-genai` SDK)
- Per-conversation history (keyed by `conversation_id`), so the model can recall earlier turns
- **Context management:** when history exceeds `MAX_HISTORY_MESSAGES` (12), the oldest messages are summarised by the LLM into a running summary that is injected into the system prompt; the last `KEEP_RECENT_MESSAGES` (6) stay verbatim. If summarisation fails, it falls back to truncation.
- System prompt defining the "Sage" persona and constraints
- Error handling: 30 s timeout, friendly messages for rate limits / auth / server / empty responses; failed turns aren't saved, so the user can simply retry
- API key from environment variable; `.env` is git-ignored
## Setup
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                              # add your key from https://aistudio.google.com/apikey
python app.py
```
Open http://localhost:5000.
 
## Testing context
1. "Hi, I'm Priya and I want to learn Python for data science."
2. Chat for 5+ turns, then ask "What was my name and goal?"
3. To see summarisation quickly, set `MAX_HISTORY_MESSAGES=4` and `KEEP_RECENT_MESSAGES=2` in `.env`, chat for several turns, and watch the server logs for `Summarised N old messages`. Then ask about your name again.
## Notes / limitations
- Conversations are stored in memory (lost on restart); swap `conversations` for Redis/SQLite for persistence.
- Token counting is approximated by message count; a production version would count tokens.

