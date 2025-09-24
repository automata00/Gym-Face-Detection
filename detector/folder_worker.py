# detector/folder_worker.py
import argparse, time, random, csv, sys, io
from pathlib import Path
import requests
from PIL import Image

def log_local(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts","camera","image","best_name","best_score","match","thr"])
        if new_file:
            w.writeheader()
        w.writerow(row)

def post_match(api_url: str, img_bytes: bytes, thr: float, camera: str) -> dict:
    files = {"file": ("frame.jpg", img_bytes, "image/jpeg")}
    params = {"thr": str(thr), "camera": camera}
    r = requests.post(f"{api_url}/match", files=files, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def post_event(api_url: str, person: str, score: float, camera: str):
    # server-side event logger (even for non-matches)
    payload = {"person": person, "score": float(score), "camera": camera}
    try:
        requests.post(f"{api_url}/events", json=payload, timeout=10)
    except requests.exceptions.RequestException:
        # non-fatal: keep going even if /events fails
        pass

def read_image_as_jpeg_bytes(path: Path) -> bytes:
    # normalize to JPEG to avoid content-type mismatches (e.g., PNG/HEIC)
    with Image.open(path) as im:
        buf = io.BytesIO()
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

def main():
    p = argparse.ArgumentParser(description="Folder → /match simulator with /events logging")
    p.add_argument("--folder", default="samples", help="Folder with images (jpg/jpeg/png)")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of your API")
    p.add_argument("--camera", default="simulator-1", help="Camera ID/name to include in logs")
    p.add_argument("--thr", type=float, default=0.45, help="Cosine similarity threshold")
    p.add_argument("--interval", type=float, default=1.0, help="Seconds between frames")
    p.add_argument("--loop", action="store_true", help="Loop forever")
    p.add_argument("--shuffle", action="store_true", help="Shuffle order each pass")
    p.add_argument("--log", default="logs/folder_worker.csv", help="Local CSV log path")
    args = p.parse_args()

    api_url = args.url.rstrip("/")
    folder = Path(args.folder)
    if not folder.exists():
        print(f"[!] Folder not found: {folder}", file=sys.stderr); sys.exit(1)

    files = []
    for ext in ("*.jpg","*.jpeg","*.png"):
        files.extend(folder.glob(ext))
    files = sorted(files)
    if not files:
        print(f"[!] No images found in {folder}", file=sys.stderr); sys.exit(1)

    print(f"▶ sending {len(files)} images from {folder} → {api_url}/match  "
          f"(thr={args.thr}, interval={args.interval}s, camera={args.camera})")
    try:
        while True:
            batch = files.copy()
            if args.shuffle:
                random.shuffle(batch)
            for img in batch:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                try:
                    img_bytes = read_image_as_jpeg_bytes(img)
                    resp = post_match(api_url, img_bytes, args.thr, args.camera)

                    if "best" in resp:
                        name = resp["best"]["name"]
                        score = float(resp["best"]["score"])
                        match = bool(resp["best"]["match"])
                        sym = "✅" if match else "❌"
                        print(f"{sym} {img.name:30s} → {name:12s}  score={score:.3f}  thr={args.thr}  cam={args.camera}")

                        # client-side CSV log
                        log_local(Path(args.log), {
                            "ts": ts, "camera": args.camera, "image": img.name,
                            "best_name": name, "best_score": f"{score:.3f}",
                            "match": "1" if match else "0", "thr": f"{args.thr:.2f}",
                        })

                        # server-side event for ALL frames:
                        post_event(api_url, name if match else "unknown", score, args.camera)
                    else:
                        print(f"⚠️  {img.name:30s} → API response had no 'best' key: {resp}")

                except requests.exceptions.RequestException as e:
                    print(f"🚫 network error for {img.name}: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"⚠️  failed on {img.name}: {e}", file=sys.stderr)
                time.sleep(args.interval)
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\n⏹ stopped by user")

if __name__ == "__main__":
    main()