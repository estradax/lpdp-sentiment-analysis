"""
FastAPI inference server for LPDP Sentiment Analysis.

Run with:
    uv run server.py

Routes:
    GET /api/inference/random-forest?text=<input text>
    GET /api/inference/indobert?text=<input text>
"""

import logging
import sys

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("lpdp_server")

app = FastAPI(
    title="LPDP Sentiment Analysis API",
    description="Inference API for LPDP Twitter sentiment analysis using Random Forest and IndoBERT models.",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Lazy-loaded model singletons
# ---------------------------------------------------------------------------
_predictors: dict = {}


def _get_predictor(model_name: str):
    """Return a cached predictor instance, loading it on first call."""
    if model_name in _predictors:
        return _predictors[model_name]

    if model_name == "random-forest":
        from src.inference import SentimentPredictor

        logger.info("Loading Random Forest model …")
        _predictors[model_name] = SentimentPredictor(model_dir="weights/random-forest")
    elif model_name == "indobert":
        from src.inference import BertSentimentPredictor

        logger.info("Loading IndoBERT model …")
        _predictors[model_name] = BertSentimentPredictor(model_dir="weights/indobert")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return _predictors[model_name]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/inference/{model_name}")
async def inference(
    model_name: str,
    text: str = Query(..., description="Input text to analyze"),
):
    """Run sentiment inference using the specified model.

    **model_name** must be one of `random-forest` or `indobert`.
    """
    if model_name not in ("random-forest", "indobert"):
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Use 'random-forest' or 'indobert'.",
        )

    if not text.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'text' must not be empty.")

    try:
        predictor = _get_predictor(model_name)
        result = predictor.predict(text)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model weights not available: {exc}",
        )
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse(content=result)


@app.get("/")
async def root():
    """Health-check / welcome endpoint."""
    return {
        "service": "LPDP Sentiment Analysis API",
        "version": "0.1.0",
        "routes": [
            "/api/inference/random-forest?text=<your text>",
            "/api/inference/indobert?text=<your text>",
        ],
    }


# ---------------------------------------------------------------------------
# Entry-point so `uv run server.py` works
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
