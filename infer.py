import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import cv2
import torch
import numpy as np
import faiss
from ultralytics import YOLO
from torchvision import transforms
from elephant_vision.reid_model import ElephantReIDModel
from elephant_vision.enhance import enhance_image
from elephant_vision.transforms import get_infer_transform
from PIL import Image

# -------- CONFIG ---------
MODEL_PATH = "artifacts/models/reid_model.pt"
INDEX_PATH = "artifacts/indexes/elephants.index"
NAMES_PATH = "artifacts/indexes/elephant_names.npy"
YOLO_PATH = "yolo11n.pt"
THRESHOLD = 0.20
OUTPUT_DIR = "artifacts/outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- INITIALIZE -------
yolo = YOLO(YOLO_PATH)
reid = ElephantReIDModel().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
missing = reid.load_state_dict(state_dict, strict=False)
reid.eval()
print("⚠️ Skipped keys:", missing.missing_keys)

index = faiss.read_index(INDEX_PATH)
names = np.load(NAMES_PATH)

transform = get_infer_transform()

def identify_frame(frame):
    results = yolo(frame)
    annotated = frame.copy()

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            if int(cls) != 20:  # class 20 = elephant
                continue

            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop = enhance_image(crop)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = Image.fromarray(crop)

            img_tensor = transform(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                embedding = reid(img_tensor).cpu().numpy().astype("float32")

            D, I = index.search(embedding, 3)
            dist, idx = D[0][0], I[0][0]
            match = max(0, 1 - dist / THRESHOLD) * 100

            match_wantted = 99 # sure

            if match >= match_wantted:
                name = names[idx]
                label = f"{name} ({match:.3f}%)"
                color = (0, 255, 0)
            else:
                name = "Unknown"
                label = f"Unknown ({match:.3f}%)"
                color = (0, 0, 255)
                
            print(f"Detected: {names[idx]} | Match: {match:.3f}% | Distance: {dist:.4f}")

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated


def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = identify_frame(frame)
        out.write(annotated)
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"🖼 Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"✅ Done! Saved to {output_path}")


def process_image(image_path, output_path=None):
    frame = cv2.imread(image_path)
    annotated = identify_frame(frame)
    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"✅ Saved annotated image to {output_path}")
    annotated = cv2.resize(annotated, None, fx=0.5, fy=0.5)
    cv2.imshow("Identified Elephant", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mode = "image"

    if mode == "video":
        video_path = r"D:\Identification_Elephant_Coaster_Center\SampleDataset\aiko.mp4"
        output_path = os.path.join(OUTPUT_DIR, "identified_test.mp4")
        process_video(video_path, output_path)
    else:
        image_path = r"C:\Users\Administrator\Desktop\Identification_Elephant_Coaster_Center\SampleDatasets\Rat_sa_mee\img_ Rat_sa_mee_right_25.png"
        output_path = os.path.join(OUTPUT_DIR, "identified_test.jpg")
        process_image(image_path, output_path)

