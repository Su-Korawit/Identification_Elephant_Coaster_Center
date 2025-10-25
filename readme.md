````markdown
# Elephant Identification System  
**ระบบระบุตัวตนช้างรายตัวแบบอัตโนมัติ**

> เอกสารนี้อธิบายสถาปัตยกรรมทั้งหมดของระบบ Elephant Vision / Elephant Identification  
> ตั้งแต่การเตรียมข้อมูล → เทรนโมเดล → สร้างฐานข้อมูลความจำ → รันทำนายช้างจากภาพหรือวิดีโอ  
> เอกสารนี้ออกแบบมาสำหรับคนที่จะดูแล/ต่อยอดระบบ ไม่ใช่แค่รันตาม


---

## ตารางสารบัญ

1. [เป้าหมายของระบบ](#เป้าหมายของระบบ)
2. [ภาพรวมการทำงาน (High-level Pipeline)](#ภาพรวมการทำงาน-high-level-pipeline)
3. [โฟลเดอร์สำคัญและ artifact ที่สร้างขึ้น](#โฟลเดอร์สำคัญและ-artifact-ที่สร้างขึ้น)
4. [สรุปบทบาทของแต่ละโมดูลโค้ด](#สรุปบทบาทของแต่ละโมดูลโค้ด)
5. [การเตรียมข้อมูล (Data Preparation Pipeline)](#การเตรียมข้อมูล-data-preparation-pipeline)
6. [การเทรนโมเดล ReID (Training Pipeline)](#การเทรนโมเดล-reid-training-pipeline)
7. [การสร้างฐานความจำ FAISS (Index Building Pipeline)](#การสร้างฐานความจำ-faiss-index-building-pipeline)
8. [การทำนาย (Inference Pipeline)](#การทำนาย-inference-pipeline)
9. [หลักการวิทยาศาสตร์เบื้องหลังระบบ](#หลักการวิทยาศาสตร์เบื้องหลังระบบ)
10. [การปรับค่า / จุดที่ควรจูนสำหรับภาคสนาม](#การปรับค่า--จุดที่ควรจูนสำหรับภาคสนาม)
11. [สรุปสุดท้ายของระบบ](#สรุปสุดท้ายของระบบ)


---

## เป้าหมายของระบบ

ระบบนี้ถูกออกแบบมาเพื่อแก้ปัญหานี้:

- ถ่ายรูป / วิดีโอของช้างในป่า ค่ายช่วยเหลือช้าง หรือศูนย์เพาะเลี้ยง
- ป้อนเข้าโปรแกรม
- โปรแกรมตอบว่า “นี่คือช้างตัวไหน” (ตัวตนเฉพาะ เช่น Aiko, Ploy, Rat_sa_mee)
- ถ้าไม่เคยเห็นมาก่อน ให้บอกว่า “Unknown” อย่างสุภาพ ไม่มั่ว

นี่ต่างจากการตรวจจับสัตว์ทั่วไปแบบ “นี่คือช้าง” มาก เพราะที่นี่เราต้องการระดับ “นี่คือช้างตัวนี้ ไม่ใช่ตัวนั้น”

> นี่คือปัญหา Re-Identification (ReID) แบบเดียวกับกล้องวงจรปิดที่ตามหาคนเดิมในอีกมุมกล้อง  
> แต่เราทำกับช้าง 🐘


---

## ภาพรวมการทำงาน (High-level Pipeline)

ลำดับขั้นทั้งหมด ตั้งแต่ไฟล์ดิบจนถึงการระบุชื่อช้างจากภาพใหม่:

```text
1. เก็บข้อมูล (วิดีโอ/ภาพจริงของช้าง)
2. yolo_crop.py
    - ใช้ YOLO ตรวจหาช้างในภาพ/วิดีโอ
    - ครอบเฉพาะตัวช้างเป็นภาพเดี่ยว
    - กรองภาพเบลอ
    - ปรับคุณภาพ
    - จัดเก็บลงโฟลเดอร์ของช้างแต่ละตัว
3. reid_dataset.py + transforms.py + enhance.py
    - โหลดรูปช้าง (โฟลเดอร์ละ 1 ตัว)
    - ทำ augmentation (เพิ่มความหลากหลาย)
    - เตรียม DataLoader
4. reid_model.py + train_reid.py
    - สร้างโมเดล ReID (ResNet-18 → embedding)
    - เทรนโมเดลให้รู้จักความต่างของแต่ละตัว
    - เซฟโมเดลที่ดีที่สุด (reid_model.pt)
5. build_faiss.py
    - ใช้โมเดลที่เทรนแล้วสร้าง embedding ของทุกภาพในฐานข้อมูล
    - สร้าง FAISS index (ฐานความจำระหว่างช้างทุกตัว)
    - เซฟ index + mapping ว่า embedding ไหนคือช้างชื่ออะไร
6. infer.py
    - รับภาพ/วิดีโอใหม่
    - ตรวจช้าง → ครอบ → ปรับ → สร้าง embedding
    - ค้นหาใน FAISS index ว่าเหมือนช้างตัวไหนที่สุด
    - ตัดสินใจว่าเป็นช้างตัวนั้น หรือ “Unknown”
    - บันทึกผลพร้อมความมั่นใจและ CSV log
````

ระบบนี้จึงเป็นสายพานต่อเนื่อง ตั้งแต่ "เก็บภาพ" ไปจนถึง "ทำนายชื่อช้าง"

---

## โฟลเดอร์สำคัญและ artifact ที่สร้างขึ้น

โครงสร้างผลลัพธ์ในโปรเจกต์ (สำคัญ):

```text
artifacts/
├── datasets/
│   └── reid/
│       └── train/
│           ├── Aiko/
│           │   ├── Aiko_0001.jpg
│           │   ├── Aiko_0002.jpg
│           │   └── ...
│           ├── Ploy/
│           │   ├── Ploy_0001.jpg
│           │   └── ...
│           └── Rat_sa_mee/
│               └── ...
│
├── models/
│   └── reid_model.pt
│       ↳ โมเดลที่เทรนแล้ว (ElephantReIDModel)
│
├── indexes/
│   ├── elephants.index
│   │   ↳ FAISS index (ฐานความจำเวกเตอร์)
│   └── elephant_names.npy
│       ↳ ชื่อช้างเรียงตามลำดับเวกเตอร์ใน index
│
├── outputs/
│   ├── identified_xxx.jpg / .mp4
│   │   ↳ รูป/วิดีโอที่มีกรอบและป้ายชื่อแปะแล้ว
│   └── identification_log.csv
│       ↳ log ว่าแต่ละภาพถูกทายเป็นใคร ความมั่นใจเท่าไหร่
│
└── plots/
    └── reid_training_curves.png
        ↳ กราฟ loss และ accuracy ระหว่างการเทรน (monitor performance)
```

รอยเท้าปลายทางของระบบคือ:

* โมเดลที่เทรนแล้ว
* ฐานข้อมูลจำช้าง (index)
* เครื่องมือทำนายแบบใช้งานได้จริง

---

## สรุปบทบาทของแต่ละโมดูลโค้ด

> นี่คือรายชื่อไฟล์หลักใน `src/elephant_vision/` และหน้าที่ของแต่ละไฟล์

### 1. `yolo_crop.py`

* Input: โฟลเดอร์ภาพ, วิดีโอ, โฟลเดอร์ใหญ่ที่มีหลายช้างย่อย
* Output: โฟลเดอร์แยกตามช้างแต่ละตัว เต็มไปด้วย crop ของร่างช้างที่ชัดและสะอาด
* ใช้ YOLO เพื่อตรวจจับช้างในรูป/เฟรมวิดีโอ
* ครอบ bounding box ของช้าง
* เลือกกรอบที่ใหญ่สุด (ถือว่าเป็นช้างหลักในภาพ)
* กรองภาพเบลอ
* เรียก `enhance_image()` เพื่อปรับคุณภาพ
* เซฟลงใน dataset โครงสร้างที่ `ImageFolder` ของ PyTorch อ่านได้

สรุป: ตัวนี้คือ “ฝ่ายจัดหาวัตถุดิบ” ของระบบ

---

### 2. `enhance.py`

* ฟังก์ชัน `enhance_image(image)`:

  * ปรับ contrast ด้วย CLAHE (แก้ปัญหาแสงไม่เท่ากัน)
  * Sharpen (เพิ่มความคมขอบผิว / texture)
  * Resize เป็นขนาดมาตรฐาน (256x256)
* ฟังก์ชัน `is_blurry(image)`:

  * ใช้ Laplacian variance เพื่อดูว่าภาพเบลอเกินไปไหม
  * ถ้าเบลอ → ทิ้ง ไม่ใช้เทรน

สรุป: “แต่งรูป” + “คัดรูปห่วยทิ้ง” ก่อนเข้าโมเดล

---

### 3. `transforms.py`

* กำหนด pipeline การแปลงภาพก่อนเข้าโมเดล
* `get_train_transform()`:

  * Resize
  * Flip
  * Color jitter (จำลองแสงไม่เหมือนกัน)
  * Normalize
  * ใช้ตอนเทรน → เพิ่มความหลากหลาย → กัน overfitting
* `get_infer_transform()`:

  * Resize + Normalize เดียวกัน แต่ไม่มี random
  * ใช้ตอน infer → ผลซ้ำได้

สรุป: “มาตรฐานภาพก่อนยื่นให้โมเดลดู”

---

### 4. `reid_dataset.py`

* โหลด dataset จากโฟลเดอร์ (ImageFolder: 1 โฟลเดอร์ = 1 ช้าง)
* split train / val
* apply transform ที่ต่างกันระหว่าง train/val
* สร้าง `DataLoader` สำหรับ PyTorch

สรุป: “ห้องเรียน” ที่จัดเตรียมข้อมูลให้โมเดลใช้เรียน

---

### 5. `reid_model.py`

* นิยามสถาปัตยกรรมโมเดล ElephantReIDModel:

  * Backbone: ResNet-18 pretrained บน ImageNet

    * ทำหน้าที่ดึง feature (ลายผิว, รูปร่าง, texture)
    * เลือก ResNet-18 แทน ResNet-50 เพื่อลดขนาดและลด overfitting ใน dataset เล็ก
  * Embedding head:

    * Linear → BatchNorm → ReLU → Dropout
    * แปลง feature ให้เป็น embedding vector ขนาด `embedding_dim` (ปกติ 512)
  * L2 normalize ที่ output:

    * ผลลัพธ์สุดท้ายคือเวกเตอร์บน hypersphere (norm = 1)
    * ใช้ง่ายกับ cosine similarity / FAISS
  * Optional classifier:

    * ตอนเทรน: มี layer fc สุดท้ายสำหรับ CrossEntropy
    * ตอน infer: ไม่ต้องใช้ classifier, เอา embedding อย่างเดียว

สรุป: “สมองจำตัวช้าง (สร้างลายนิ้วมือเชิงคณิตศาสตร์ของช้าง)”

---

### 6. `train_reid.py`

งานหลักในไฟล์นี้:

* โหลด DataLoader จาก `reid_dataset.py`
* สร้างโมเดลจาก `reid_model.py`
* ตั้ง loss (CrossEntropy)
* ตั้ง optimizer (AdamW + weight decay)

  * AdamW เลือกแทน Adam เพื่อได้ L2 regularization จริง (ลด overfitting)
* ใช้ scheduler (ReduceLROnPlateau):

  * ถ้า val_loss ไม่ดีขึ้น → ลด learning rate อัตโนมัติ
* วัดทั้ง train loss / val loss / train acc / val acc
* ใช้ Early Stopping:

  * ถ้า val_loss ไม่ดีขึ้นนานเกิน patience → หยุดก่อน
  * เก็บ state_dict ของโมเดลที่ดีที่สุด
* เซฟ:

  * `artifacts/models/reid_model.pt`
  * กราฟเทรน `artifacts/plots/reid_training_curves.png`

สรุป: “ครูฝึกโมเดล”
มันสอนให้โมเดลจำความต่างระหว่างช้างแต่ละตัว

---

### 7. `build_faiss.py`

หน้าที่:

* โหลดโมเดล ReID ที่เทรนแล้ว (`reid_model.pt`)
* วนผ่าน dataset อีกครั้ง
* สร้าง embedding ของทุกภาพ (โดยโมเดล ReID)
* สร้าง FAISS index ด้วยเวกเตอร์ทั้งหมด

  * เราใช้ `faiss.IndexHNSWFlat` (กราฟ HNSW)

    * ทำให้ค้นหาเพื่อนบ้านที่คล้ายได้เร็วมาก
    * ดีพอสำหรับหลักพัน-หมื่นตัว
* เก็บชื่อช้างตามลำดับ embedding
* เซฟ 2 อย่าง:

  * index → `artifacts/indexes/elephants.index`
  * mapping ชื่อ → `artifacts/indexes/elephant_names.npy`

สรุป: “สร้างฐานความทรงจำของระบบ”
หลังขั้นนี้ ระบบมีหน่วยความจำ: เวกเตอร์ → ชื่อช้าง

---

### 8. `infer.py`

นี่คือระยะ “ใช้งานจริงในสนาม”

สิ่งที่มันทำ:

1. โหลดทุกอย่าง:

   * YOLO (ตรวจจับช้างในภาพ)
   * โมเดล ReID (สร้าง embedding ของช้าง)
   * FAISS index (ฐานจำช้าง)
   * รายชื่อช้าง (ชื่อจริงแต่ละตัว)

2. ทำงานกับ input 3 โหมด:

   * ภาพเดี่ยว
   * วิดีโอ
   * ทั้งโฟลเดอร์ (หลายโฟลเดอร์ย่อย = หลายตัวช้างทดสอบ)

3. สำหรับแต่ละภาพ/เฟรม:

   * ใช้ YOLO หา bounding box ของช้าง
   * เลือกกรอบที่ใหญ่สุด (ถ้าเปิด BIGGEST_BOX_ONLY)
   * ครอบเฉพาะช้าง
   * enhance (เพิ่ม contrast, sharpen)
   * resize, normalize
   * ป้อนเข้า ReID → ได้ embedding
   * ค้นหา embedding ใกล้สุดใน FAISS index
   * วัดระยะ (distance)
   * ถ้า `distance < THRESHOLD` → ให้ชื่อช้าง

     * วาดกรอบสีเขียว
   * ถ้าไม่ → “Unknown”

     * วาดกรอบสีแดง

4. ทำ logging:

   * เขียน CSV (`identification_log.csv`) เพื่อวิเคราะห์คุณภาพ
   * เซฟภาพ/วิดีโอที่ annotate แล้ว (แปะชื่อกับเปอร์เซ็นต์)

5. รายงานผลสรุปเป็น text ต่อโฟลเดอร์ทดสอบ:

   * ทายว่า "โฟลเดอร์นี้คือช้างตัวไหน"
   * ให้ความมั่นใจสูงสุด
   * เวลาที่ใช้

สรุป: “สมองปฏิบัติการภาคสนาม”
ตั้งแต่ภาพป่าโผล่มา → ได้ชื่อช้าง พร้อมหลักฐานและความมั่นใจ

---

## การเตรียมข้อมูล (Data Preparation Pipeline)

### เป้าหมาย

* ได้รูปช้างที่ชัดเจน สะอาด ไม่เบลอ ขนาดคงที่
* แยกเป็นโฟลเดอร์รายช้าง เพื่อป้อนให้โมเดล ReID เทรน

### เกิดขึ้นที่ไหน

* `yolo_crop.py`
* `enhance.py`

### ขั้นตอน

1. อ่าน input:

   * โฟลเดอร์ที่มีหลาย subfolder (หลายช้าง)
   * โฟลเดอร์รูปของช้างตัวเดียว
   * วิดีโอหมุนรอบตัวช้าง
2. ใช้ YOLO เพื่อหาช้างในเฟรม
3. เลือก bounding box

   * ส่วนใหญ่เราเลือก bounding box ที่ใหญ่ที่สุด
   * เชื่อว่าตัวใหญ่สุด = ตัวหลักในเฟรมนั้น
4. ครอบภาพให้อยู่แค่ตัวช้าง
5. ตัดภาพเบลอทิ้ง (`is_blurry`)
6. enhance ภาพ (`enhance_image`)
7. เซฟลงที่
   `artifacts/datasets/reid/train/<ช้างแต่ละตัว>/*.jpg`

### ทำไมต้อง Enhance?

* เพิ่ม contrast ด้วย CLAHE → สู้กับแสงกลางวัน/เย็น/เงา
* Sharpen → ดึงรายละเอียดรอยย่น, รอยแผล, texture ผิว
* Resize → ทำให้ทุกภาพมีขนาด input คงที่สำหรับโมเดล

---

## การเทรนโมเดล ReID (Training Pipeline)

### เป้าหมาย

* สอนให้โมเดล “จำความแตกต่างของช้างแต่ละตัว”
* เป้าหมายไม่ใช่การบอกว่า “นี่คือช้าง” แต่ “นี่คือช้างตัวไหน”

### เกิดขึ้นที่ไหน

* `reid_dataset.py`
* `transforms.py`
* `reid_model.py`
* `train_reid.py`

### กระบวนการ

1. โหลด dataset แบบ ImageFolder

   * โฟลเดอร์ละ 1 class = 1 ช้าง
2. split train / val
3. ทำ augmentation หนักใน train:

   * RandomResizedCrop
   * Flip ซ้ายขวา
   * ColorJitter
   * Rotation / Perspective distortion
   * GaussianBlur
     → จำลองสถานการณ์จริง: กล้องมุมแปลก, แสงต่างเวลา, สั่น, ถ่ายไกล
4. Validation ใช้ transform เบาๆ (แค่ Resize + Normalize)
   → เพื่อวัด performance จริงแบบคงที่
5. โมเดล:

   * Backbone = ResNet-18 pretrained ImageNet

     * เลือก ResNet-18 ไม่ใช่ ResNet-50 เพื่อลด overfitting (dataset ยังเล็ก)
   * เปลี่ยนหัวสุดท้ายเป็น embedding layer + BatchNorm + Dropout
   * normalize output เป็นเวกเตอร์หน่วย (L2 normalize)
6. เทรนด้วย CrossEntropyLoss

   * classifier layer สุดท้ายบอกว่าเฟรมนี้เป็นช้างตัวไหน
   * (softmax training, supervised)
7. ใช้ Early Stopping / Scheduler เพื่อลด overfit:

   * early stop ถ้า val loss ไม่ดีขึ้น
   * ลด learning rate อัตโนมัติเมื่อเริ่มตัน
8. เซฟโมเดลที่ดีที่สุดเป็น `artifacts/models/reid_model.pt`
9. วาดกราฟ loss/accuracy และเก็บใน `artifacts/plots/reid_training_curves.png`

หมายเหตุสำคัญ:

* ตอน infer เราไม่ใช้ softmax classifier สุดท้าย
* เราใช้ embedding vector ที่ normalized แล้ว
* นั่นคือ “หน้าตัวตนเชิงคณิตศาสตร์” ของช้าง

---

## การสร้างฐานความจำ FAISS (Index Building Pipeline)

### เป้าหมาย

* สร้างคลังความจำของช้างแต่ละตัวเพื่อใช้ค้นหาแบบเร็ว
* ไม่ต้อง re-train โมเดลทุกครั้งที่เจอช้างใหม่

### เกิดขึ้นที่ไหน

* `build_faiss.py`

### กระบวนการ

1. โหลดโมเดลที่เทรนแล้ว (`reid_model.pt`)
2. วนผ่าน dataset อีกครั้ง
3. สร้าง embedding ของทุกภาพด้วยโมเดล ReID
4. สร้าง FAISS index:

   * เราใช้ `faiss.IndexHNSWFlat(dim, m)`
   * นี่คือกราฟเพื่อนบ้าน (approximate nearest neighbor / ANN)
   * แลกความแม่นยำเล็กน้อยเพื่อความเร็วมหาศาล
5. บันทึก:

   * index ลงไฟล์ `elephants.index`
   * รายชื่อช้าง (`elephant_names.npy`) เรียงตาม embeddings ใน index

นี่คือ “ความจำถาวรของระบบ”
มันทำให้ตอนทำนายเราสามารถถามคำถามว่า
“เวกเตอร์นี้เหมือนเวกเตอร์ไหนในความจำที่สุด?” ได้ทันที

---

## การทำนาย (Inference Pipeline)

### เป้าหมาย

* รับภาพ/วิดีโอ/โฟลเดอร์ใหม่ แล้วบอกว่า “ช้างตัวไหน?”

### เกิดขึ้นที่ไหน

* `infer.py`

### กระบวนการใน `infer.py`

1. โหลดทั้งหมด:

   * YOLO (ตรวจจับช้าง)
   * โมเดล ReID (ดึง embedding)
   * FAISS index (ความจำช้าง)
   * `elephant_names.npy` (id → ชื่อช้าง)
2. โหมดการทำงาน:

   * `process_image()` → ภาพเดี่ยว
   * `process_video()` → วิดีโอ 1 ไฟล์
   * `process_folder()` → ทั้ง dataset (หลายโฟลเดอร์ของช้าง)
3. ในแต่ละภาพ:

   * ตรวจจับ bounding boxes ด้วย YOLO
   * เลือก bounding box ใหญ่สุด (ถ้า config บังคับ)
   * ครอบเฉพาะช้าง
   * enhance + resize + normalize
   * ป้อนเข้า ReID → ได้ embedding
   * ค้นหา k ใกล้สุดใน FAISS (k=3)
   * ดูระยะ (distance)
   * ถ้า `< THRESHOLD` → ถือว่าใช่ช้างตัวนั้น, แสดงชื่อเป็นเขียว
   * ถ้า `>= THRESHOLD` → แสดง “Unknown” แดง
4. วาดกรอบ + label ลงในภาพ/วิดีโอ
5. เขียน log ลง CSV:

   * image path
   * ชื่อที่ทำนาย
   * ความมั่นใจ (%)
   * ระยะห่างเชิงเวกเตอร์
   * หมายเหตุ เช่น Unknown หรือ mismatch ระหว่าง label กับ folder
6. พิมพ์สรุปเป็น block ต่อแต่ละโฟลเดอร์ เช่น:

   * โฟลเดอร์ Rat_sa_mee  → predict = Rat_sa_mee (99.12%)
   * ใช้เวลา 0.73s

ผลลัพธ์สุดท้ายได้ทั้ง:

* ไฟล์รูปที่ annotate แล้ว
* วิดีโอที่ annotate แล้ว
* CSV วิเคราะห์คุณภาพการระบุชื่อ

---

## หลักการวิทยาศาสตร์เบื้องหลังระบบ

### 1. YOLO (Object Detection)

* YOLO มองภาพทั้งหมดแล้วคืน bounding box ของ "ช้าง"
* เราไม่สนใจสิ่งอื่น
* ทุกอย่างเริ่มที่ bounding box ที่ถูกต้อง

ถ้า YOLO ตรวจพลาด → ทุกอย่างหลังจากนี้พัง
ดังนั้นคุณภาพ YOLO สำคัญมาก (ควร fine-tune กับช้างจริงด้วย)

---

### 2. ReID (Re-identification)

* ไม่ใช่การบอกประเภท (class = “elephant”)
* แต่คือการบอกตัวบุคคล (“นี่คือช้างหมายเลข 7 ที่เราเคยเห็น”)
* โมเดลไม่ output ป้ายชื่อโดยตรงตอน infer
* โมเดล output "embedding vector" ซึ่งเป็นรหัสตัวตน

embedding ของช้างตัวเดียวกันควรอยู่ใกล้กันใน space เดียวกัน
embedding ของช้างต่างตัวกันควรห่างกัน

---

### 3. FAISS Index

* สมองมนุษย์มีฮิปโปแคมปัสเก็บความจำระยะยาว
* FAISS คือเวอร์ชันคณิตศาสตร์ของสิ่งนั้น
* เราป้อน embedding ใหม่ → FAISS บอกว่า “คล้าย embedding ของใครมากสุด”
* เร็วมาก (ออกแบบมาสำหรับการค้นหาระยะทางใกล้สุดในเวกเตอร์ high-dimensions)

ประโยชน์ใหญ่:

* อยากเพิ่มช้างตัวใหม่?

  * เก็บภาพมัน → สร้าง embedding → เพิ่มเข้า index
  * ไม่จำเป็นต้อง retrain โมเดลทันที

---

### 4. Threshold Decision

เรามีกฎง่าย ๆ:

* ถ้าระยะ (distance) จากช้างที่ใกล้ที่สุดยังเกิน `THRESHOLD` → เราบอกว่า "Unknown"

นี่ป้องกันโมเดลพูดมั่ว เช่น ช้างใหม่ที่ไม่เคยเห็น = "Ploy"
ระบบจะสุภาพ บอกว่า "Unknown" แทน ซึ่งเป็นพฤติกรรมที่ถูกต้องสำหรับงานภาคสนาม

ค่าของ THRESHOLD ต้องจูนตาม dataset ของจริง:

* ต่ำเกินไป → จะบอก Unknown ตลอด
* สูงเกินไป → จะเดามั่วเป็นช้างที่อยู่ใกล้ที่สุดเสมอ

การจูน threshold = การจูนบุคลิกของระบบ

---

## การปรับค่า / จุดที่ควรจูนสำหรับภาคสนาม

1. `frame_stride` ใน `yolo_crop.py`

   * เก็บทุกกี่เฟรมจากวิดีโอ
   * ถ้าสูง → เก็บน้อยลง ประหยัดเวลา
   * ถ้าต่ำ → เก็บเยอะขึ้น ได้ความหลากหลายมากขึ้น

2. `is_blurry` threshold ใน `enhance.py`

   * ใช้กรองภาพไม่ชัด
   * ถ้าค่าต่ำไป → ภาพเบลอหลุดมาเยอะ โมเดลเรียนแต่ภาพขยะ
   * ถ้าค่าสูงไป → อาจเผลอทิ้งภาพที่ยังพอใช้ได้

3. Augmentation strength ใน `reid_dataset.py` / `transforms.py`

   * เพิ่ม ColorJitter / Rotation ฯลฯ
   * ยิ่ง augmentation เยอะ → โมเดลจำ “ตัวตน” มากกว่าท่าทาง
   * แต่ถ้ามากเกินไปจนข้อมูลเพี้ยน → โมเดลจะเรียน noise

4. `embedding_dim` ใน `reid_model.py`

   * ปกติใช้ 512
   * ถ้าขยายเป็น 1024 → มีข้อมูลมากขึ้นต่อหนึ่งตัวช้าง แต่ index ก็ใหญ่ขึ้น
   * ถ้าลดเหลือ 256 → เบาลง เร็วขึ้น ประหยัดหน่วยความจำ

5. `THRESHOLD` ใน `infer.py`

   * ตัวตัดสิน Unknown/ไม่ Unknown
   * ควบคุมนิสัยของระบบว่าจะ "มั่นใจและพูดชื่อ" หรือ "ปลอดภัยและสุภาพ"

---

## สรุปสุดท้ายของระบบ

นี่คือหน้าที่ของระบบ Elephant Identification แบบสั้นและตรง:

* `yolo_crop.py`: ดึงตัวช้างออกมาจากวิดีโอ/ภาพ และเตรียม dataset ที่สะอาด
* `enhance.py` + `transforms.py`: ทำให้ภาพได้มาตรฐานและคุณภาพดี
* `reid_dataset.py`: เตรียมข้อมูลเป็น batch พร้อม augmentation
* `reid_model.py`: โมเดลที่แปลงภาพช้าง → embedding ที่แทน “ตัวตน”
* `train_reid.py`: ฝึกโมเดลให้จำความต่างของช้างแต่ละตัว
* `build_faiss.py`: สร้างฐานข้อมูลความจำของช้างทุกตัว (FAISS index)
* `infer.py`: ตอนใช้งานจริง — เห็นช้างใหม่แล้วบอกได้ว่าใคร หรือไม่รู้จัก
* `artifacts/`: ที่เก็บผลลัพธ์ทั้งหมด (โมเดล, index, กราฟ, log, output ที่ annotate)

พูดอีกแบบ:

1. ระบบ “มองเห็นช้าง”
2. ระบบ “จำหน้าช้าง”
3. ระบบ “ค้นหาว่าหน้านี้เหมือนใครที่สุด”
4. ระบบ “ตัดสินใจและอธิบายให้มนุษย์เข้าใจได้ทันที พร้อมหลักฐาน”

หรือในเวอร์ชันหนึ่งประโยค:

> เราสร้างให้คอมพิวเตอร์ทำในสิ่งที่คนเลี้ยงช้างทำมานาน — จำตัวช้างจากแผลเก่า ลายผิว และรูปทรงร่าง — แล้วใส่มันเข้าไปในโค้ด.

```
```
