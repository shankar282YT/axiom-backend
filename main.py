# ============================================================
#  AXIOM · main.py
#  FastAPI server
#  POST /axiom  { "message": "..." }  →  { "reply": "..." }
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import nlp
import engine

# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title       = "AXIOM API",
    description = "Hybrid NLP + Rule-based AI backend",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Schemas ──────────────────────────────────────────────────
class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)

class MessageResponse(BaseModel):
    reply:      str
    intent:     str
    confidence: float

class HealthResponse(BaseModel):
    status:  str
    version: str

# ── Routes ───────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
def health():
    return {"status": "online", "version": "1.0.0"}


@app.post("/axiom", response_model=MessageResponse)
def axiom_reply(req: MessageRequest):
    try:
        # 1. NLP — understand intent
        intent_json = nlp.predict(req.message)

        # 2. Rule engine — pick response
        reply = engine.execute(intent_json)

        return {
            "reply":      reply,
            "intent":     intent_json["intent"],
            "confidence": intent_json["confidence"],
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
