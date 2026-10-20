"""
Minimal HTTP API example (FastAPI). The models are loaded once at startup (CXR_MODELS, default
all of them; VRAM per model in README); requests are served one at a time because a single
GPU generates one batch at a time.

    pip install fastapi==0.115.12 uvicorn==0.34.2 python-multipart==0.0.20
    uvicorn serve_example:app --host 0.0.0.0 --port 8000
    CXR_MODELS=medgemma uvicorn serve_example:app ...        # only one model (Windows cmd: set CXR_MODELS=medgemma)

    curl http://localhost:8000/models
    curl -X POST http://localhost:8000/report -F "file=@cxr.png" -F "view=PA" -F "model=chexone"
    -> {"model": "chexone", "findings": "...", "impression": "...", "view": "PA", "seconds": 9.8, ...}
    curl -X POST http://localhost:8000/report -F "file=@cxr.png" -F "view=PA" -F "model=all"
    -> {"reports": {"medgemma": {...}, "chexone": {...}}, "view": "PA", ...}
"""

import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from cxr_report import MODELS, ReportGenerator, load_image
from cxr_report.prompts import normalize_view

DISCLAIMER = ("Draf laporan otomatis dari model riset. Bukan diagnosis; wajib ditinjau dan "
              "disetujui oleh dokter spesialis radiologi.")
state = {}
gpu_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    keys = [k.strip() for k in os.environ.get("CXR_MODELS", ",".join(MODELS)).split(",") if k.strip()]
    state["generators"] = {key: ReportGenerator(key) for key in keys}
    yield
    state.clear()


app = FastAPI(title="Indonesian CXR report generation", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "models": {k: g.info for k, g in state["generators"].items()}}


@app.get("/models")
def models():
    return [{"model": k, "name": MODELS[k].name, "license": MODELS[k].license} for k in state["generators"]]


@app.post("/report")
async def report(file: UploadFile = File(...), view: str | None = Form(None), model: str | None = Form(None)):
    loaded = state["generators"]
    model = (model or next(iter(loaded))).strip().lower()
    if model != "all" and model not in loaded:
        raise HTTPException(422, f"model {model!r} is not loaded; use one of {[*loaded, 'all']}")
    try:
        view = normalize_view(view)
    except ValueError as e:
        raise HTTPException(422, str(e))
    try:
        image = load_image(await file.read())
    except Exception:
        raise HTTPException(400, "file is not a readable image (PNG/JPEG; export DICOM first)")

    def run():
        with gpu_lock:
            t0 = time.time()
            keys = list(loaded) if model == "all" else [model]
            res = {key: loaded[key].generate(image, view) for key in keys}
            return res, time.time() - t0

    res, seconds = await run_in_threadpool(run)
    meta = {"view": view, "seconds": round(seconds, 1), "disclaimer": DISCLAIMER}
    if model == "all":
        return {"reports": res, **meta}
    return {"model": model, **res[model], **meta}
