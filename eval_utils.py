# eval_utils.py
# ฟังก์ชันสรุปผลแบบ pure Python (ไม่ต้องใช้ torch/faiss) ใช้กับ evaluate_by_view.py
import re
from collections import defaultdict

VIEWS = ("front", "back", "left", "right")
_VIEW_RE = re.compile(r"(?:^|[_\-\s])(front|back|left|right)(?:[_\-\s.]|$)", re.IGNORECASE)


def parse_view(filename: str) -> str:
    """อ่านมุมกล้องจากชื่อไฟล์ เช่น img_Rat_sa_mee_right_25.png -> 'right'"""
    m = _VIEW_RE.search(filename)
    return m.group(1).lower() if m else "unknown_view"


def summarize(rows, thresholds):
    """
    rows: list ของ dict {"true": ชื่อจริง, "view": มุม, "pred": ชื่อที่ใกล้ที่สุดใน index, "dist": ระยะ}
    thresholds: list ของค่า distance ที่ใช้ตัดสิน (dist <= t ถึงจะตอบชื่อ ไม่งั้นตอบ Unknown)
    คืน list ของ dict ต่อ (threshold, view) พร้อม n, correct, wrong, unknown และสัดส่วน
    """
    results = []
    for t in thresholds:
        stats = defaultdict(lambda: {"n": 0, "correct": 0, "wrong": 0, "unknown": 0})
        for r in rows:
            for key in (r["view"], "ALL"):
                s = stats[key]
                s["n"] += 1
                if r["dist"] > t:
                    s["unknown"] += 1
                elif r["pred"] == r["true"]:
                    s["correct"] += 1
                else:
                    s["wrong"] += 1
        for view, s in sorted(stats.items(), key=lambda kv: (kv[0] == "ALL", kv[0])):
            n = s["n"]
            results.append({
                "threshold": t,
                "view": view,
                "n": n,
                "correct": s["correct"],
                "wrong": s["wrong"],
                "unknown": s["unknown"],
                "correct_pct": round(100 * s["correct"] / n, 2) if n else 0.0,
                "wrong_pct": round(100 * s["wrong"] / n, 2) if n else 0.0,
                "unknown_pct": round(100 * s["unknown"] / n, 2) if n else 0.0,
            })
    return results
