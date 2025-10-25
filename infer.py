import sys, os, csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

# -------- FEATURE FLAGS ---------
EXPORT_CSV = True                    # เปิด/ปิด export csv logs
ONLY_UNKNOWN_NOT_LAST = True         # export เฉพาะ unknown ที่ไม่ใช่ folder สุดท้าย
EXPORT_MIS_MATCH = True              # export กรณีชื่อไม่ตรงกับชื่อ folder
BIGGEST_BOX_ONLY = True              # ใช้เฉพาะ bounding box ที่ใหญ่ที่สุด

CSV_PATH = os.path.join(OUTPUT_DIR, "identification_log.csv")

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

# -------- CSV INIT ---------
if EXPORT_CSV:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ImagePath", "DetectedName", "Match%", "Distance", "FolderName", "Note"])


def log_csv(image_path, name, match, dist, note=""):
    if not EXPORT_CSV:
        return
    folder_name = os.path.basename(os.path.dirname(image_path))
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([image_path, name, f"{match:.3f}", f"{dist:.4f}", folder_name, note])


def identify_frame(frame, image_path=None):
    results = yolo(frame)
    annotated = frame.copy()

    # รวมทุก box แล้วหาอันใหญ่สุด (ถ้าเปิด)
    all_boxes, all_classes = [], []
    for r in results:
        all_boxes.extend(r.boxes.xyxy.cpu().numpy())
        all_classes.extend(r.boxes.cls.cpu().numpy())

    if BIGGEST_BOX_ONLY and len(all_boxes) > 0:
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in all_boxes]
        max_idx = int(np.argmax(areas))
        all_boxes = [all_boxes[max_idx]]
        all_classes = [all_classes[max_idx]]

    for box, cls in zip(all_boxes, all_classes):
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

        match_wantted = 99
        if match >= match_wantted:
            name = names[idx]
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        label = f"{name} ({match:.3f}%)"
        print(f"Detected: {name} | Match: {match:.3f}% | Distance: {dist:.4f}")

        # ---------- CSV LOG ----------
        if EXPORT_CSV and image_path:
            folder_name = os.path.basename(os.path.dirname(image_path))
            not_last = len(os.listdir(os.path.dirname(image_path))) > 0
            if (ONLY_UNKNOWN_NOT_LAST and name == "Unknown" and not_last) or \
               (EXPORT_MIS_MATCH and name != "Unknown" and name != folder_name):
                note = "Unknown" if name == "Unknown" else f"Mismatch ({folder_name})"
                log_csv(image_path, name, match, dist, note)

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
    annotated = identify_frame(frame, image_path)
    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"✅ Saved annotated image to {output_path}")
    annotated = cv2.resize(annotated, None, fx=0.5, fy=0.5)
    cv2.imshow("Identified Elephant", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_folder(folder_path, output_base=OUTPUT_DIR):
    os.makedirs(output_base, exist_ok=True)
    print(f"📁 Starting identification in folder: {folder_path}")

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, folder_path)
                out_dir = os.path.join(output_base, os.path.dirname(rel_path))
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"identified_{file}")

                try:
                    process_image(input_path, out_path)
                except Exception as e:
                    print(f"❌ Error processing {file}: {e}")

    print("✅ Folder identification completed.")


if __name__ == "__main__":
    mode = "folder"  # "image", "video", หรือ "folder"

    if mode == "video":
        video_path = r"D:\Identification_Elephant_Coaster_Center\SampleDataset\aiko.mp4"
        output_path = os.path.join(OUTPUT_DIR, "identified_test.mp4")
        process_video(video_path, output_path)

    elif mode == "image":
        image_path = r"C:\Users\Administrator\Desktop\Identification_Elephant_Coaster_Center\SampleDatasets\Rat_sa_mee\img_Rat_sa_mee_right_25.png"
        output_path = os.path.join(OUTPUT_DIR, "identified_test.jpg")
        process_image(image_path, output_path)

    elif mode == "folder":
        folder_path = r"C:\Users\Administrator\Desktop\Identification_Elephant_Coaster_Center\SampleDatasets"
        process_folder(folder_path)
