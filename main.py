
from datetime import date
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="Local LLM Invoice Extractor")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "gemma3:1b-it-qat"


class InvoiceRequest(BaseModel):
    text: str = Field(min_length=1)


class InvoiceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor: str
    amount: float
    currency: Literal["USD", "EUR", "GBP"]
    date: str


@app.get("/")
def root():
    return {"status": "ok", "service": "invoice-extractor"}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/extract", response_model=InvoiceResult)
async def extract_invoice(request: InvoiceRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Empty invoice text")

    prompt = f"""
Extract the invoice vendor, total amount due, currency and payment due date.
Return JSON only with keys vendor, amount, currency, date.
The amount must be numeric. Currency must be USD, EUR, or GBP.
Date must be YYYY-MM-DD. Do not invent values.

Invoice:
{text}
"""

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": InvoiceResult.model_json_schema(),
                    "options": {"temperature": 0},
                },
            )
            response.raise_for_status()
            result = InvoiceResult.model_validate_json(
                response.json()["response"]
            )
            date.fromisoformat(result.date)
            return result
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama unavailable")
    except (httpx.HTTPError, ValueError, KeyError):
        raise HTTPException(status_code=502, detail="Invoice extraction failed")
