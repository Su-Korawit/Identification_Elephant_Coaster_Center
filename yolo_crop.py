# src/elephant_vision/yolo_crop.py
import os
import cv2
from ultralytics import YOLO
from .enhance import enhance_image, is_blurry

def process_input(input_path, elephant_name, frame_stride=5):
    """ครอบช้างจากภาพ, วิดีโอ, หรือโฟลเดอร์ แล้วบันทึกลง dataset"""
    model = YOLO("yolo11n.pt")

    output_dir = f"artifacts/datasets/reid/train/{elephant_name}"
    os.makedirs(output_dir, exist_ok=True)

    # --- ถ้าเป็นโฟลเดอร์ของภาพ ---
    if os.path.isdir(input_path):
        image_files = [f for f in os.listdir(input_path)
                       if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        saved = 0
        skip_stride = frame_stride  # ใช้ข้ามทุก N ภาพ

        for i, img_name in enumerate(sorted(image_files)):
            # 🔹 ข้ามภาพตาม stride เช่น ทุก 5 ภาพ
            if i % skip_stride != 0:
                continue

            img_path = os.path.join(input_path, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            results = model(frame, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            if len(boxes) == 0:
                continue

            # เอากล่องที่ใหญ่สุด
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
            max_idx = int(areas.index(max(areas)))
            x1, y1, x2, y2 = map(int, boxes[max_idx][:4])

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or is_blurry(crop):
                continue

            enhanced = enhance_image(crop)
            save_path = os.path.join(output_dir, f"{elephant_name}_{i:04d}.jpg")
            cv2.imwrite(save_path, enhanced)
            saved += 1

        print(f"✅ Saved {saved} images (every {skip_stride}th) from folder to {output_dir}")
        return

    # --- ถ้าเป็นวิดีโอ ---
    if input_path.lower().endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(input_path)
        frame_idx = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 🔹 ข้ามเฟรมตาม stride เช่น ทุก 5 เฟรม
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            results = model(frame, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            if len(boxes) == 0:
                frame_idx += 1
                continue

            # เอากล่องที่ใหญ่สุด
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
            max_idx = int(areas.index(max(areas)))
            x1, y1, x2, y2 = map(int, boxes[max_idx][:4])

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or is_blurry(crop):
                frame_idx += 1
                continue

            enhanced = enhance_image(crop)
            save_path = os.path.join(output_dir, f"{elephant_name}_{frame_idx:04d}.jpg")
            cv2.imwrite(save_path, enhanced)
            saved += 1
            frame_idx += 1

        cap.release()
        print(f"✅ Saved {saved} images (every {frame_stride}th frame) from video to {output_dir}")
