"""Operator app.   uvicorn app.server:app --host 0.0.0.0 --port 8000

Configuration by environment variables (so the same code runs on a laptop and on the Pi):
  DETECTOR_WEIGHTS  runs/neu_y11n/weights/best.pt   or   best_executorch_model/
  DEFECT_DIR        data/neu_yolo/images/test
  CLEAN_DIR         data/clean_test            (optional)
  CAMERA            set to 0 to use a real camera instead of the conveyor replay
  MODBUS_HOST/PORT  127.0.0.1 / 5020
  ADVISOR           llama | template
  LLAMA_RUNNER / LLAMA_MODEL / LLAMA_TOKENIZER   paths for llama_main, the .pte and tokenizer.model"""
import os
import threading

import cv2
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from app.db import DB
from app.pipeline import Pipeline
from app.sources import CameraSource, ConveyorSim
from config import CFG
from advisor.llama_runner import LlamaAdvisor, template_advice
from spikeguard_twin.modbus_tcp import MachineStateReader
from vision.detector import Detector

app = FastAPI(title="SpikeGuard-QC")
STATE = {}


class TemplateAdvisor:
    def advise(self, ctx):
        return template_advice(ctx), {"source": "template", "wall_ms": 0}


@app.on_event("startup")
def startup():
    env = os.environ.get
    source = (CameraSource(int(env("CAMERA"))) if env("CAMERA") is not None
              else ConveyorSim(env("DEFECT_DIR", "data/neu_yolo/images/test"), env("CLEAN_DIR"), 0.1))
    advisor = (LlamaAdvisor(env("LLAMA_RUNNER"), env("LLAMA_MODEL"), env("LLAMA_TOKENIZER"))
               if env("ADVISOR", "template") == "llama" else TemplateAdvisor())
    pipe = Pipeline(Detector(env("DETECTOR_WEIGHTS", "runs/neu_y11n/weights/best.pt")), source,
                    MachineStateReader(env("MODBUS_HOST", "127.0.0.1"), int(env("MODBUS_PORT", "5020")),
                                       CFG.fusion.stale_after_s),
                    advisor, DB(env("DB_PATH", "inspections.db")))
    STATE["pipe"] = pipe
    threading.Thread(target=pipe.run, kwargs={"fps": float(env("FPS", "5"))}, daemon=True).start()


@app.get("/")
def index():
    return FileResponse("app/static/index.html")


@app.get("/api/status")
def status():
    p = STATE["pipe"]
    return {"status": p.latest_status, "advice": p.latest_advice}


@app.get("/api/inspections")
def inspections(n: int = 20):
    return STATE["pipe"].db.recent("inspections", n)


@app.get("/api/advisories")
def advisories(n: int = 10):
    return STATE["pipe"].db.recent("advisories", n)


class Scenario(BaseModel):
    defect_prob: float


@app.post("/api/scenario")
def scenario(s: Scenario):
    """Demo control: change how often defective parts arrive (conveyor replay only)."""
    src = STATE["pipe"].source
    if hasattr(src, "defect_prob"):
        src.defect_prob = max(0.0, min(1.0, s.defect_prob))
    return {"defect_prob": getattr(src, "defect_prob", None)}


@app.get("/api/frame.jpg")
def frame():
    f = STATE["pipe"].latest_frame
    if f is None:
        return Response(status_code=204)
    ok, buf = cv2.imencode(".jpg", f)
    return Response(buf.tobytes(), media_type="image/jpeg")


@app.get("/api/export")
def export():
    n = STATE["pipe"].db.export_csv("inspections", "inspections_export.csv")
    return {"rows": n, "file": "inspections_export.csv"}
