import numpy as np, cv2
from pathlib import Path
from insightface.app import FaceAnalysis

# Two images to compare (put real face photos here)
IMG1 = Path("samples/entry.jpg")
IMG2 = Path("samples/entry2.jpg")

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def embed(app: FaceAnalysis, p: Path) -> np.ndarray:
    if not p.exists():
        raise SystemExit(f"Missing image: {p}")
    img = cv2.imread(str(p))
    if img is None:
        raise SystemExit(f"Could not read {p} (convert to real JPEG?)")
    faces = app.get(img)
    if not faces:
        raise SystemExit(f"No face detected in {p}")
    return faces[0].embedding.astype("float32")

def main():
    # Load RetinaFace + ArcFace on CPU
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1)

    e1 = embed(app, IMG1)
    e2 = embed(app, IMG2)
    s = cosine(e1, e2)

    print(f"Cosine similarity = {s:.3f}")
    thr = 0.45  # start here; tune with your own data
    print("Match?", "YES" if s >= thr else "NO", f"(threshold={thr})")

if __name__ in ("__main__", "<run_path>"):
    main()