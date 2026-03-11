from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import nlp
import engine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    message: str

@app.get("/")
def root():
    return {"message": "AXIOM backend is running"}

@app.post("/axiom")
def axiom(query: Query):
    try:
        # Step 1 — NLP understands the message, returns intent JSON
        intent_json = nlp.predict(query.message)

        # Step 2 — Rule engine picks the right response
        reply = engine.execute(intent_json)

        return {"reply": reply}

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run `python train.py` first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
