# gym_api/app.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import cv2
import json, io, csv

from insightface.app import FaceAnalysis

# ───────────────────────────────────────────────────────────────
# FastAPI app
# ───────────────────────────────────────────────────────────────
app = FastAPI(title="Gym Face Detection API")

# ───────────────────────────────────────────────────────────────
# Events log (in-memory) + Debounce + Persistence
# ───────────────────────────────────────────────────────────────
EVENTS: list[dict] = []

# Debounce: one event per (person,camera) within this many seconds
DEDUPE_WINDOW_SEC = 30

# Tracks last time we logged each (person, camera)
LAST_SEEN: dict[tuple[str, str], float] = {}

# Persistence path
EVENTS_PATH = Path("data/events.json")
EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_events():
    global EVENTS
    if EVENTS_PATH.exists():
        try:
            EVENTS = json.loads(EVENTS_PATH.read_text())
        except Exception:
            EVENTS = []
    else:
        EVENTS = []

def save_events():
    EVENTS_PATH.write_text(json.dumps(EVENTS, indent=2))

class Event(BaseModel):
    person: str
    score: float
    camera: str = "simulator"
    timestamp: str | None = None

def _should_log(person: str, camera: str, ts_iso: str | None = None) -> bool:
    """
    Return True if we should log a new event for (person, camera),
    i.e., last event is older than DEDUPE_WINDOW_SEC. Updates LAST_SEEN on allow.
    """
    now = datetime.fromisoformat(ts_iso) if ts_iso else datetime.now(timezone.utc)
    key = (person, camera)
    last_ts = LAST_SEEN.get(key)
    if last_ts is not None:
        delta = (now.timestamp() - last_ts)
        if delta < DEDUPE_WINDOW_SEC:
            return False
    LAST_SEEN[key] = now.timestamp()
    return True

# Load any existing events on startup
load_events()

# ───────────────────────────────────────────────────────────────
# Face model
# ───────────────────────────────────────────────────────────────
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(640, 640))

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype("float32"); b = b.astype("float32")
    an = a / (np.linalg.norm(a) + 1e-9)
    bn = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(an, bn))

def embed_image_bytes(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    faces = face_app.get(img)
    if not faces:
        raise ValueError("No face detected")
    # choose largest face
    def area(f):
        x1, y1, x2, y2 = map(int, f.bbox)
        return max(0, x2-x1) * max(0, y2-y1)
    f_big = max(faces, key=area)
    return f_big.embedding.astype("float32")

# ───────────────────────────────────────────────────────────────
# Gallery storage (JSON file)
# ───────────────────────────────────────────────────────────────
GALLERY_PATH = Path("data/embeddings/gallery.json")
GALLERY_PATH.parent.mkdir(parents=True, exist_ok=True)

def read_gallery() -> dict[str, list[float]]:
    if not GALLERY_PATH.exists():
        return {}
    return json.loads(GALLERY_PATH.read_text())

def write_gallery(d: dict):
    GALLERY_PATH.write_text(json.dumps(d, indent=2))

# ───────────────────────────────────────────────────────────────
# Routes
# ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/enroll")
async def enroll(name: str = "", files: list[UploadFile] = File(...)):
    if not name:
        return JSONResponse(
            {"detail":[{"type":"missing","loc":["body","name"],"msg":"Field required","input":None}]},
            status_code=422
        )
    gal = read_gallery()
    embs = []
    for uf in files:
        b = await uf.read()
        embs.append(embed_image_bytes(b))
    if not embs:
        return JSONResponse({"error":"no usable images"}, status_code=400)
    avg = np.mean(np.stack(embs, axis=0), axis=0).astype("float32")
    gal[name] = avg.tolist()
    write_gallery(gal)
    return {"enrolled": name, "images": len(embs), "people": list(gal.keys())}

@app.post("/match")
async def match(file: UploadFile = File(...), thr: float = 0.45, camera: str = "simulator"):
    gal = read_gallery()
    if not gal:
        return JSONResponse({"error":"gallery empty, enroll first"}, status_code=400)
    b = await file.read()
    try:
        q = embed_image_bytes(b)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    scores = {name: cosine(np.array(vec, dtype="float32"), q) for name, vec in gal.items()}
    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]
    is_match = best_score >= thr

    # Debounced auto-log for matches (with persistence)
    logged = False
    if is_match:
        ts_iso = datetime.utcnow().isoformat()
        if _should_log(best_name, camera, ts_iso=ts_iso):
            EVENTS.append({
                "person": best_name,
                "score": round(best_score, 3),
                "camera": camera,
                "timestamp": ts_iso
            })
            save_events()
            logged = True

    return {
        "scores": {k: round(v,3) for k,v in scores.items()},
        "best": {"name": best_name, "score": round(best_score,3), "match": is_match},
        "thr": thr,
        "logged": logged
    }

@app.get("/events")
def list_events():
    # latest first for convenience
    return {"events": list(reversed(EVENTS))}

@app.post("/events")
def add_event(ev: Event):
    ts = ev.timestamp or datetime.utcnow().isoformat()
    if _should_log(ev.person, ev.camera, ts_iso=ts):
        EVENTS.append({
            "person": ev.person,
            "score": round(float(ev.score), 3),
            "camera": ev.camera,
            "timestamp": ts
        })
        save_events()
        return {"status": "logged", "count": len(EVENTS)}
    else:
        return {"status": "skipped_duplicate", "count": len(EVENTS)}

@app.post("/events/clear")
def clear_events():
    global EVENTS
    EVENTS = []
    save_events()
    return {"status": "cleared"}

@app.get("/events/csv")
def events_csv():
    # stream a CSV from current EVENTS
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["timestamp","camera","person","score"])
    writer.writeheader()
    for e in EVENTS:
        writer.writerow({
            "timestamp": e.get("timestamp",""),
            "camera": e.get("camera",""),
            "person": e.get("person",""),
            "score": e.get("score",""),
        })
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=events.csv"}
    )