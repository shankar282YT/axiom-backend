from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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
    return {"reply": f"AXIOM received: {query.message}"}
