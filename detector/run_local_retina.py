# detector/run_local_retina.py
import sys, os, csv
from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis

IMG = Path("samples/entry.jpg")
OUT_ANN = Path("samples/entry_retina_detected.jpg")
OUT_DIR = Path("data/faces")
EMB_DIR = Path("data/embeddings")

def main():
    print("▶ run_local_retina.py starting")
    print("• CWD:", os.getcwd())
    print("• IMG exists?", IMG.exists())

    # initialize face analysis
    app = FaceAnalysis(name="buffalo_l")   # <-- letter L, not number 1
    app.prepare(ctx_id=-1)                 # CPU only
    print("✅ FaceAnalysis ready")

    if not IMG.exists():
        sys.exit("❌ Missing samples/entry.jpg")
    img = cv2.imread(str(IMG))
    if img is None:
        sys.exit("❌ OpenCV could not read samples/entry.jpg")

    faces = app.get(img)
    print(f"✅ Detected {len(faces)} face(s)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    h, w = img.shape[:2]
    embs, meta_rows = [], []

    for i, f in enumerate(faces):
        x1, y1, x2, y2 = map(int, f.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # save crop
        y1c, y2c = max(0, y1), min(h, y2)
        x1c, x2c = max(0, x1), min(w, x2)
        crop = img[y1c:y2c, x1c:x2c]
        if crop.size:
            crop_path = OUT_DIR / f"retina_face_{i:02d}.jpg"
            cv2.imwrite(str(crop_path), crop)
            print("• wrote crop:", crop_path)

        # collect embedding
        if getattr(f, "embedding", None) is not None and f.embedding.size:
            emb = f.embedding.astype("float32")
            embs.append(emb)
            np.save(EMB_DIR / f"emb_{i:02d}.npy", emb)
            meta_rows.append({
                "idx": i,
                "det_score": float(getattr(f, "det_score", 0.0))
            })

    # save annotated image
    cv2.imwrite(str(OUT_ANN), img)
    print("✅ wrote annotated:", OUT_ANN)

    # save embeddings and metadata
    if embs:
        embs = np.vstack(embs)
        np.save(EMB_DIR / "embeddings.npy", embs)
        with (EMB_DIR / "meta.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["idx", "det_score"])
            writer.writeheader()
            writer.writerows(meta_rows)
        print("✅ wrote embeddings:", EMB_DIR / "embeddings.npy", "shape:", embs.shape)
        print("✅ wrote metadata:", EMB_DIR / "meta.csv")
    else:
        print("ℹ️ no embeddings produced")

# run when script is executed directly or via runpy
if __name__ in ("__main__", "<run_path>"):
    main()