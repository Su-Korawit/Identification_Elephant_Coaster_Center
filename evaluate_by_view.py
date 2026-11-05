# evaluate_by_view.py
# วัดความแม่นยำของระบบระบุตัวช้างแยกตามมุมกล้อง (front / back / left / right)
#
# วิธีใช้ (วางไฟล์นี้ข้างๆ infer.py และรันจากโฟลเดอร์เดียวกัน):
#   python evaluate_by_view.py --root path/to/QUERY_DATASET
#
# โครงสร้างโฟลเดอร์ที่คาดหวัง:
#   QUERY_DATASET/<ชื่อช้าง>/img_<ชื่อช้าง>_<front|back|left|right>_<เลข>.png
#
# สำคัญ: QUERY_DATASET ต้องเป็นภาพที่ "ไม่ได้ใช้สร้าง FAISS index"
# ถ้าใช้ภาพชุดเดียวกับที่ build index ผลจะสูงเกินจริง (ระยะ ~0 กับตัวเอง)
import sys, os, csv, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import torch
import faiss
from PIL import Image
from ultralytics import YOLO

from elephant_vision.reid_model import ElephantReIDModel
from elephant_vision.enhance import enhance_image
from elephant_vision.transforms import get_infer_transform
from eval_utils import parse_view, summarize

MODEL_PATH = "artifacts/models/reid_model.pt"
INDEX_PATH = "artifacts/indexes/elephants.index"
NAMES_PATH = "artifacts/indexes/elephant_names.npy"
YOLO_PATH = "yolo11n.pt"
ELEPHANT_CLASS = 20  # COCO class 20 = elephant (เหมือน infer.py)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_system():
    yolo = YOLO(YOLO_PATH)
    reid = ElephantReIDModel().to(DEVICE)
    reid.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    reid.eval()
    index = faiss.read_index(INDEX_PATH)
    names = np.load(NAMES_PATH)
    return yolo, reid, index, names, get_infer_transform()


def nearest(frame, yolo, reid, index, names, transform):
    """คืน (ชื่อที่ใกล้ที่สุด, ระยะ) หรือ None ถ้าตรวจไม่เจอช้าง"""
    boxes = []
    for r in yolo(frame, verbose=False):
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            if int(c) == ELEPHANT_CLASS:
                boxes.append(b)
    if not boxes:
        return None
    x1, y1, x2, y2 = map(int, max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = enhance_image(crop)
    crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        emb = reid(transform(crop).unsqueeze(0).to(DEVICE)).cpu().numpy().astype("float32")
    D, I = index.search(emb, 1)
    return str(names[I[0][0]]), float(D[0][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="โฟลเดอร์ query: <root>/<ชื่อช้าง>/*.png|jpg")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.002, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0],
                    help="ค่า distance ที่จะลอง (infer.py ตอนนี้เทียบเท่า ~0.002)")
    ap.add_argument("--out", default="artifacts/outputs/eval_by_view.csv")
    args = ap.parse_args()

    yolo, reid, index, names, transform = load_system()
    rows, no_detect = [], {}
    for true_name in sorted(os.listdir(args.root)):
        folder = os.path.join(args.root, true_name)
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            frame = cv2.imread(os.path.join(folder, fn))
            if frame is None:
                continue
            view = parse_view(fn)
            res = nearest(frame, yolo, reid, index, names, transform)
            if res is None:
                no_detect[view] = no_detect.get(view, 0) + 1
                continue
            rows.append({"true": true_name, "view": view, "pred": res[0], "dist": res[1]})

    if not rows:
        print("ไม่มีภาพที่ตรวจพบช้างเลย ตรวจ --root และโฟลเดอร์ artifacts/")
        return

    results = summarize(rows, args.thresholds)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print(f"\nภาพที่ตรวจพบช้าง: {len(rows)} | ตรวจไม่พบช้างแยกตามมุม: {no_detect or 'ไม่มี'}")
    print(f"{'thr':>6} {'view':>13} {'n':>5} {'correct%':>9} {'wrong%':>7} {'unknown%':>9}")
    for r in results:
        print(f"{r['threshold']:>6} {r['view']:>13} {r['n']:>5} {r['correct_pct']:>9} {r['wrong_pct']:>7} {r['unknown_pct']:>9}")
    print(f"\nบันทึกผลที่ {args.out}")


if __name__ == "__main__":
    main()
