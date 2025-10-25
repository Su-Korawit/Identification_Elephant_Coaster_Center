# src/elephant_vision/yolo_crop.py
import os
import cv2
from ultralytics import YOLO
from .enhance import enhance_image, is_blurry

def process_input(input_path, elephant_name=None, frame_stride=5):
    """ครอบช้างจากภาพ, วิดีโอ, หรือโฟลเดอร์ (รวม subfolder) แล้วบันทึกลง dataset"""
    model = YOLO("yolo11n.pt")

    base_output = "artifacts/datasets/reid/train"
    os.makedirs(base_output, exist_ok=True)

    # --- 📦 ถ้าเป็นโฟลเดอร์หลักที่มีหลายช้าง ---
    if os.path.isdir(input_path):
        subfolders = [os.path.join(input_path, d) for d in os.listdir(input_path)
                      if os.path.isdir(os.path.join(input_path, d))]
        if len(subfolders) > 0:
            print(f"📁 Detected {len(subfolders)} subfolders. Processing all elephants...")
            for sub in subfolders:
                name = os.path.basename(sub)
                print(f"\n🐘 Processing elephant: {name}")
                process_input(sub, name, frame_stride=frame_stride)
            print("✅ All subfolders processed successfully.")
            return

    # --- 🖼 ถ้าเป็นโฟลเดอร์ของภาพ (ไม่มี subfolder ย่อย) ---
    if os.path.isdir(input_path):
        if elephant_name is None:
            elephant_name = os.path.basename(os.path.normpath(input_path))

        output_dir = os.path.join(base_output, elephant_name)
        os.makedirs(output_dir, exist_ok=True)

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

        print(f"✅ Saved {saved} images (every {skip_stride}th) from {input_path} → {output_dir}")
        return

    # --- 🎞 ถ้าเป็นวิดีโอ ---
    if input_path.lower().endswith((".mp4", ".avi", ".mov")):
        if elephant_name is None:
            elephant_name = os.path.splitext(os.path.basename(input_path))[0]

        output_dir = os.path.join(base_output, elephant_name)
        os.makedirs(output_dir, exist_ok=True)

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
        print(f"✅ Saved {saved} frames (every {frame_stride}th) from video {input_path} → {output_dir}")
