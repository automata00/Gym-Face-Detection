# Gym Face Detection and Access Control

This project provides a face detection and recognition system for gym entrances.  
It uses FastAPI, InsightFace, and RetinaFace/YOLO, and is designed to run locally or on a Jetson Orin Nano.

---

## Features
- Face detection and recognition using embeddings
- Enroll members with one or more images
- Match incoming images to enrolled members
- Event logging with debounce to prevent duplicate entries
- CSV export of logs
- FastAPI interface with Swagger UI for testing
- Designed to integrate with access control (door relay or maglock)

---

## Project Structure

```
gym-face-detection/
├─ gym_api/
│  └─ app.py            # FastAPI app (enroll, match, events)
├─ detector/
│  ├─ folder_worker.py  # Simulates camera by sending images
│  └─ run_local_retina.py
├─ recognizer/
│  ├─ enroll.py
│  ├─ match.py
│  └─ verify_pair.py
├─ data/
│  ├─ embeddings/       # saved embeddings and gallery
│  └─ events.json       # logged events
├─ samples/             # demo images
├─ logs/                # logs and csv files
├─ requirements.txt
└─ README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the API
```bash
uvicorn gym_api.app:app --reload --host 0.0.0.0 --port 8000
```

- Health check: http://127.0.0.1:8000/health  
- Docs: http://127.0.0.1:8000/docs  

### 3. Enroll a member
```bash
curl -X POST "http://127.0.0.1:8000/enroll" \
  -F "name=marc" \
  -F "files=@samples/marc_1.jpg" \
  -F "files=@samples/marc_2.jpg"
```

### 4. Match a face
```bash
curl -X POST "http://127.0.0.1:8000/match?thr=0.45&camera=door-1" \
  -F "file=@samples/entry.jpg"
```

### 5. Check events
```bash
curl http://127.0.0.1:8000/events
curl http://127.0.0.1:8000/events/csv -o events.csv
```

---

## How It Works

- Each face image is converted into a 512-dimensional vector (embedding).
- Matching is done using cosine similarity between embeddings.
- A threshold value determines whether a match is accepted.
- Events are logged with a short debounce window to prevent duplicate entries.

---

## Roadmap
- Add RTSP worker for live camera streams
- Integrate door relay for access control
- Add authentication and dashboard
- Optimize for Jetson Orin Nano
