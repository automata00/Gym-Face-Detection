# recognizer/match.py
import argparse, json, numpy as np, cv2
from pathlib import Path
from insightface.app import FaceAnalysis

DB = Path("data/embeddings/gallery.json")

def norm(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32"); return v / (np.linalg.norm(v) + 1e-9)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(norm(a), norm(b)))

def embed(app: FaceAnalysis, img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None: raise SystemExit(f"Cannot read {img_path}")
    faces = app.get(img)
    if not faces: raise SystemExit(f"No face in {img_path}")
    return faces[0].embedding.astype("float32")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", help="query image path (e.g. samples/entry2.jpg)")
    p.add_argument("--thr", type=float, default=0.45, help="cosine threshold for match")
    args = p.parse_args()

    if not DB.exists(): raise SystemExit(f"Missing {DB}. Enroll first.")
    gallery = json.loads(DB.read_text())
    if not gallery: raise SystemExit("Empty gallery. Enroll at least one person.")

    app = FaceAnalysis(name="buffalo_l"); app.prepare(ctx_id=-1)
    q = embed(app, Path(args.image))

    scores = {name: cosine(np.array(vec, dtype="float32"), q) for name, vec in gallery.items()}
    best = max(scores, key=scores.get)
    print("scores:", {k: round(v,3) for k,v in scores.items()})
    print(f"best: {best} ({scores[best]:.3f})")
    print("match?", "YES" if scores[best] >= args.thr else "NO", f"(thr={args.thr})")

if __name__ in ("__main__", "<run_path>"):
    main()