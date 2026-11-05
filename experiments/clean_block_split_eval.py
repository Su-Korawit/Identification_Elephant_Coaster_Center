# clean_exp.py - ทดลองแบบแบ่งข้อมูลตามบล็อกเฟรม (กัน leakage จากเฟรมติดกัน) แล้ววัดผลแยกตามมุม
import sys, os, re, json, random, time
PROJ = r"D:\Ai Hackathon Elephant\Production D-Day\Identification_Elephant_Coaster_Center"
SCR = r"C:\Users\Admin\AppData\Local\Temp\claude\bab25de5585b923e\b75b8b34-e6c6-491b-9676-d09d472fbf5f\scratchpad"
sys.path.insert(0, os.path.join(PROJ, "src")); sys.path.insert(0, os.path.join(SCR, "eleph"))
WORK = os.path.join(SCR, "exp"); os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)

import cv2, numpy as np, torch, torch.nn as nn, faiss
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from elephant_vision.reid_model import ElephantReIDModel
from elephant_vision.enhance import enhance_image
from eval_utils import parse_view

DEV = "cuda"
DATA = os.path.join(PROJ, "Day1_workshop_data_E")
CROPS = os.path.join(WORK, "crops"); os.makedirs(CROPS, exist_ok=True)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
random.seed(0); np.random.seed(0); torch.manual_seed(0)

# ---------- Phase A: crop cache ----------
meta_path = os.path.join(WORK, "meta.json")
if os.path.exists(meta_path):
    meta = json.load(open(meta_path, encoding="utf-8"))
else:
    yolo = YOLO(os.path.join(PROJ, "yolo11n.pt"))
    meta, nodet = [], {}
    for name in sorted(os.listdir(DATA)):
        out = os.path.join(CROPS, name.strip()); os.makedirs(out, exist_ok=True)
        files = sorted(f for f in os.listdir(os.path.join(DATA, name)) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        t0 = time.time()
        for k, f in enumerate(files):
            frame = cv2.imread(os.path.join(DATA, name, f))
            if frame is None: continue
            view = parse_view(f); m = re.search(r"_(\d+)\.\w+$", f); n = int(m.group(1)) if m else -1
            boxes = []
            for r in yolo(frame, verbose=False):
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                    if int(c) == 20: boxes.append(b)
            if not boxes:
                nodet[(name.strip(), view)] = nodet.get((name.strip(), view), 0) + 1; continue
            x1, y1, x2, y2 = map(int, max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1])))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            p = os.path.join(out, f"{view}_{n}.jpg"); cv2.imwrite(p, enhance_image(crop))
            meta.append({"name": name.strip(), "view": view, "n": n, "path": p})
            if k % 500 == 0: print("crop", name, k, len(files), f"{time.time()-t0:.0f}s", flush=True)
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump({f"{a}|{b}": v for (a, b), v in nodet.items()}, open(os.path.join(WORK, "nodet.json"), "w"))
    print("NODET", nodet, flush=True)
print("crops", len(meta), flush=True)

# ---------- Phase B: split ตามบล็อกเฟรม ----------
split = {"train": [], "val": [], "test": []}
cells = {}
for m in meta: cells.setdefault((m["name"], m["view"]), []).append(m)
for (nm, vw), L in cells.items():
    L.sort(key=lambda m: m["n"]); N = len(L)
    for i, m in enumerate(L):
        r = i / N
        if r < 0.60: split["train"].append(m)
        elif 0.63 <= r < 0.73: split["val"].append(m)
        elif r >= 0.76: split["test"].append(m)   # ช่วง 0.60-0.63 และ 0.73-0.76 ทิ้ง กันเฟรมติดกัน
classes = sorted({m["name"] for m in meta}); cidx = {c: i for i, c in enumerate(classes)}
print({k: len(v) for k, v in split.items()}, classes, flush=True)
tcount = {}
for m in split["test"]: tcount[(m["name"], m["view"])] = tcount.get((m["name"], m["view"]), 0) + 1
print("test per cell", {f"{a}|{b}": c for (a, b), c in sorted(tcount.items())}, flush=True)

norm = transforms.Normalize([0.5]*3, [0.5]*3)
tr_tf = transforms.Compose([transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0), ratio=(0.8, 1.2)), transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4), transforms.RandomRotation(15), transforms.RandomPerspective(0.3, 0.5),
    transforms.GaussianBlur(3), transforms.ToTensor(), norm])
ev_tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), norm])

class DS(Dataset):
    def __init__(s, items, tf): s.items, s.tf = items, tf
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        m = s.items[i]; return s.tf(Image.open(m["path"]).convert("RGB")), cidx[m["name"]], i

def loader(items, tf, shuffle): return DataLoader(DS(items, tf), batch_size=32, shuffle=shuffle, num_workers=0)

model = ElephantReIDModel(num_classes=len(classes)).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-4); ce = nn.CrossEntropyLoss()

@torch.no_grad()
def embed(items):
    model.eval(); E = []
    for x, y, _ in loader(items, ev_tf, False):
        e, _ = model(x.to(DEV)); E.append(e.cpu().numpy())
    return np.concatenate(E).astype("float32")

@torch.no_grad()
def val_loss_acc():
    model.eval(); tl = c = n = 0
    for x, y, _ in loader(split["val"], ev_tf, False):
        _, lg = model(x.to(DEV)); y = y.to(DEV)
        tl += ce(lg, y).item() * len(y); c += (lg.argmax(1) == y).sum().item(); n += len(y)
    return tl / n, 100 * c / n

def evaluate_test():
    Etr = embed(split["train"]); idx = faiss.IndexFlatL2(Etr.shape[1]); idx.add(Etr)
    names_tr = [m["name"] for m in split["train"]]
    Ete = embed(split["test"]); D, I = idx.search(Ete, 1)
    rows = [{"true": m["name"], "view": m["view"], "pred": names_tr[I[k][0]], "dist": float(D[k][0])} for k, m in enumerate(split["test"])]
    return rows

hist, snaps = [], {}
for ep in range(1, EPOCHS + 1):
    model.train(); tl = c = n = 0
    for x, y, _ in loader(split["train"], tr_tf, True):
        x, y = x.to(DEV), y.to(DEV); _, lg = model(x); loss = ce(lg, y)
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item() * len(y); c += (lg.argmax(1) == y).sum().item(); n += len(y)
    vl, va = val_loss_acc()
    h = {"epoch": ep, "train_loss": tl / n, "train_acc": 100 * c / n, "val_loss": vl, "val_acc": va}
    hist.append(h); print(h, flush=True)
    if ep in (1, 3, 5, 10, 15, 20, EPOCHS):
        snaps[ep] = evaluate_test()
        acc = {}
        for r in snaps[ep]:
            for k in (r["view"], "ALL"):
                a = acc.setdefault(k, [0, 0]); a[0] += 1; a[1] += int(r["pred"] == r["true"])
        print("  TEST@", ep, {k: round(100 * v[1] / v[0], 1) for k, v in sorted(acc.items())}, flush=True)
    if vl == min(x["val_loss"] for x in hist): best_ep = ep; torch.save(model.state_dict(), os.path.join(WORK, "best_val.pt"))
print("best_val_epoch", best_ep, flush=True)
model.load_state_dict(torch.load(os.path.join(WORK, "best_val.pt"), map_location=DEV))
snaps["best"] = evaluate_test()
json.dump({"hist": hist, "snaps": {str(k): v for k, v in snaps.items()}, "best_ep": best_ep, "classes": classes,
           "counts": {k: len(v) for k, v in split.items()}, "test_cells": {f"{a}|{b}": c for (a, b), c in tcount.items()}},
          open(os.path.join(WORK, "result.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("DONE", flush=True)
