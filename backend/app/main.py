"""
FairLens AI — FastAPI Backend
Run with: uvicorn app.main:app --reload
"""
from app.services.explainer import explain
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.services.model_service import predict

# ── Create FastAPI app ─────────────────────────────────────────────────────
app = FastAPI(
    title="FairLens AI",
    description="Bias detection API using DistilBERT",
    version="1.0"
)

# ── Allow React frontend to talk to this API ───────────────────────────────
# Without this, the browser blocks requests from a different port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server address
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response shapes ────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str          # the text the user wants to analyze

class AnalyzeResponse(BaseModel):
    label:      str    # "Biased" or "Not Biased"
    confidence: float  # e.g. 0.91
    scores:     dict   # {"Biased": 0.91, "Not Biased": 0.09}


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """ Health check — visit http://localhost:8000 to confirm API is running """
    return {"status": "FairLens AI is running!"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(request: AnalyzeRequest):
    """
    Main endpoint — takes text, returns bias prediction.

    Example request body:
    { "text": "Women are bad at driving." }

    Example response:
    { "label": "Biased", "confidence": 0.93, "scores": {...} }
    """
    result = predict(request.text)
    return result
class ExplainResponse(BaseModel):
    text:        str
    label:       str
    confidence:  float
    scores:      dict
    explanation: list     # list of word importance dicts


@app.post("/explain", response_model=ExplainResponse)
def explain_text(request: AnalyzeRequest):
    """
    Same as /analyze but also returns which words caused the bias.

    Example response:
    {
      "text": "Women are bad at driving.",
      "label": "Biased",
      "confidence": 0.91,
      "scores": {"Not Biased": 0.09, "Biased": 0.91},
      "explanation": [
        {"word": "Women", "score": 0.32, "direction": "biased"},
        {"word": "bad",   "score": 0.28, "direction": "biased"},
        {"word": "driving", "score": 0.09, "direction": "biased"}
      ]
    }
    """
    result = explain(request.text)
    return result