from __future__ import annotations

import os
import tempfile
from typing import Annotated

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from openai_service import extract_invoice_data

app = FastAPI(title="Image OCR Demo")
# app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})


@app.post("/extract-text")
async def extract_text(file: Annotated[UploadFile, File(...)], purpose: Annotated[str, Form()] = "ocr"):
    if not file.filename:
        return JSONResponse(status_code=400, content={"error": "Please choose an image file."})

    suffix = os.path.splitext(file.filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        temp_path = tmp.name

    try:
        if os.getenv("OPENAI_API_KEY"):
            invoice_json = extract_invoice_data(temp_path)
        else:
            invoice_json = {"error": "OPENAI_API_KEY is not set"}

        return {
            "invoice_json": invoice_json,
            "filename": file.filename,
            "purpose": purpose,
        }
    except Exception as exc:  # pragma: no cover - defensive path
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
