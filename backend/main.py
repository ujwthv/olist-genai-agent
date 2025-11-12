import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
import agents  # after env

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
def chat(req: ChatRequest):
    return agents.handle_user_message(req.message, req.history)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug-env")
def debug_env():
    return {
        "has_gemini": bool(os.getenv("GEMINI_API_KEY")),
        "has_openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
        "openrouter_model": os.getenv("OPENROUTER_MODEL"),
        "gemini_model": os.getenv("GEMINI_MODEL"),
    }

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"type": "error", "error": str(exc)})
