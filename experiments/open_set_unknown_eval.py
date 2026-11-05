import sys, os, json, glob
PROJ = r"D:\Ai Hackathon Elephant\Production D-Day\Identification_Elephant_Coaster_Center"
SCR = r"C:\Users\Admin\AppData\Local\Temp\claude\bab25de5585b923e\b75b8b34-e6c6-491b-9676-d09d472fbf5f\scratchpad"
sys.path.insert(0, os.path.join(PROJ, "src")); os.chdir(os.path.join(SCR, "exp"))
import cv2, numpy as np, torch, faiss
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from elephant_vision.reid_model import ElephantReIDModel
from elephant_vision.enhance import enhance_image
meta = json.load(open("meta.json", encoding="utf-8"))
res = json.load(open("result.json", encoding="utf-8")); classes = res["classes"]
m = ElephantReIDModel(num_classes=len(classes)).to("cuda"); m.load_state_dict(torch.load("best_val.pt", map_location="cuda")); m.eval()
tf = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])
def emb(img):
    with torch.no_grad(): e,_ = m(tf(img).unsqueeze(0).to("cuda")); return e.cpu().numpy().astype("float32")
# index จากชุด train (60% แรกตามบล็อก เหมือน clean_exp)
cells = {}
for x in meta: cells.setdefault((x["name"], x["view"]), []).append(x)
tr = []
for k, L in cells.items():
    L.sort(key=lambda x: x["n"]); N = len(L); tr += [x for i, x in enumerate(L) if i / N < 0.60]
E = np.concatenate([emb(Image.open(x["path"]).convert("RGB")) for x in tr]); idx = faiss.IndexFlatL2(E.shape[1]); idx.add(E)
names = [x["name"] for x in tr]
yolo = YOLO(os.path.join(PROJ, "yolo11n.pt")); out = []
for p in sorted(glob.glob(os.path.join(PROJ, "Unknown", "*", "*.*"))):
    fr = cv2.imread(p); bx = []
    for r in yolo(fr, verbose=False):
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            if int(c) == 20: bx.append(b)
    if not bx: out.append((os.path.basename(p), "nodet")); continue
    x1,y1,x2,y2 = map(int, max(bx, key=lambda b:(b[2]-b[0])*(b[3]-b[1]))); cr = enhance_image(fr[y1:y2, x1:x2])
    D, I = idx.search(emb(Image.fromarray(cv2.cvtColor(cr, cv2.COLOR_BGR2RGB))), 1)
    out.append((os.path.basename(p), names[I[0][0]], round(float(D[0][0]), 3)))
for o in out: print(o)
