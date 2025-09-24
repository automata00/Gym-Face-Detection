import csv
from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ------ Paths ------
IMG = Path("samples/entry.jpg")  # Put a test image here
OUT_ANN = Path("samples/entry_retina_detected.jpg")
OUT_DIR = Path("data/faces")                 # face crops
EMB_DIR = Path("data/embeddings")            # embeddings + metadata

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def main():
    # Load RetinaFace + ArcFace (downloads on first run to ~/.insightface)
    app = FaceAnalysis(name="buffalo_l")   # <-- letter L (large model pack)
    app.prepare(ctx_id=-1)                 # CPU only (Mac/Windows/Linux)

    # Read input image
    if not IMG.exists():
        raise SystemExit(f"Missing test image: {IMG}. Put a face photo at this path.")
    img = cv2.imread(str(IMG))
    if img is None:
        raise SystemExit(f"Could not read {IMG}")

    # Detect faces
    faces = app.get(img)
    print(f"Detected {len(faces)} face(s)")

    # Ensure output dirs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    # Draw boxes, save crops, collect embeddings
    h, w = img.shape[:2]
    embs = []
    meta_rows = []
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = map(int, f.bbox)

        # draw bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{float(f.det_score):.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # save crop
        y1c, y2c = max(0, y1), min(h, y2)
        x1c, x2c = max(0, x1), min(w, x2)
        crop = img[y1c:y2c, x1c:x2c]
        if crop.size:
            crop_path = OUT_DIR / f"retina_face_{i:02d}.jpg"
            cv2.imwrite(str(crop_path), crop)

        # collect embedding (512-D)
        if getattr(f, "embedding", None) is not None and f.embedding.size:
            emb = f.embedding.astype("float32")
            embs.append(emb)
            # also save per-face embedding if you want
            np.save(EMB_DIR / f"emb_{i:02d}.npy", emb)
            meta_rows.append({
                "idx": i,
                "det_score": float(f.det_score),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })

    # Save annotated image
    cv2.imwrite(str(OUT_ANN), img)
    print(f"Saved annotated image -> {OUT_ANN}")

    # Save stacked embeddings + metadata
    if embs:
        embs = np.vstack(embs)  # (N, 512)
        np.save(EMB_DIR / "embeddings.npy", embs)
        with (EMB_DIR / "meta.csv").open("w", newline="") as fh:
            wcsv = csv.DictWriter(fh, fieldnames=["idx", "det_score", "x1", "y1", "x2", "y2"])
            wcsv.writeheader()
            wcsv.writerows(meta_rows)
        print(f"Saved {embs.shape[0]} embedding(s) -> {EMB_DIR/'embeddings.npy'}")
        print(f"Saved metadata -> {EMB_DIR/'meta.csv'}")
        if embs.shape[0] >= 2:
            print("cosine(emb0, emb1) =", round(cosine(embs[0], embs[1]), 3))
    else:
        print("No embeddings produced (no face or model pack missing recognition).")

    print(f"Saved crops -> {OUT_DIR}/")

if __name__ == "__main__":
    main()