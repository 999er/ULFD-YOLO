import os
import numpy as np
from collections import defaultdict

# ============================== 1. 你的真实路径配置 ==============================
# 真实标签路径 (你的数据集 val labels)
GT_DIR        = "/home/featurize/work/yolov13-main/datasets/names/labels/val"              

# 你的模型预测结果路径 (你之前跑出来的那个成绩单)
PRED_A        = "/home/featurize/work/runs/detect/val_FishBJ_Bootstrap/labels"   

# YOLOv11n 原版预测结果路径 (【注意】这里一定要填你刚刚用 best.pt 跑出来的那个最新 labels 路径！)
PRED_B        = "/home/featurize/work/runs/detect/val_yolov11n/labels"    

NAME_A, NAME_B = "ULFD-YOLO", "YOLOv11n"
N_CLASSES     = 21
N_BOOTSTRAP   = 1000
SEED          = 42
IOU_50        = [0.5]
IOU_COCO      = [round(0.5 + 0.05 * i, 2) for i in range(10)]
# ==============================================================================

# ============================== 2. 我帮你补全的核心代码 ==============================
def load_ground_truth(gt_dir):
    """读取真实标签，并将 cx, cy, w, h 转换为 x1, y1, x2, y2"""
    gt_dict = {}
    if not os.path.exists(gt_dir): return gt_dict
    for file in os.listdir(gt_dir):
        if not file.endswith('.txt') or file == 'classes.txt': continue
        img_id = file.split('.')[0]
        boxes = []
        with open(os.path.join(gt_dir, file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c = float(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    x1, y1 = cx - w/2, cy - h/2
                    x2, y2 = cx + w/2, cy + h/2
                    boxes.append([c, x1, y1, x2, y2])
        gt_dict[img_id] = np.array(boxes)
    return gt_dict

def load_predictions(pred_dir):
    """读取预测标签，包含置信度，并将 cx, cy, w, h 转换为 x1, y1, x2, y2"""
    pred_dict = {}
    if not os.path.exists(pred_dir): return pred_dict
    for file in os.listdir(pred_dir):
        if not file.endswith('.txt') or file == 'classes.txt': continue
        img_id = file.split('.')[0]
        boxes = []
        with open(os.path.join(pred_dir, file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    c = float(parts[0])
                    cx, cy, w, h, conf = map(float, parts[1:6])
                    x1, y1 = cx - w/2, cy - h/2
                    x2, y2 = cx + w/2, cy + h/2
                    boxes.append([c, conf, x1, y1, x2, y2])
        pred_dict[img_id] = np.array(boxes)
    return pred_dict
# =======================================================================================

def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)

def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

def compute_map(image_ids, gt, pred, iou_thrs, n_classes):
    aps = []
    for thr in iou_thrs:
        cls_ap = []
        for c in range(n_classes):
            scores, tps, n_gt = [], [], 0
            for k, img in enumerate(image_ids):
                g = gt.get(img, np.zeros((0, 5)))
                p = pred.get(img, np.zeros((0, 6)))
                g = g[g[:, 0] == c][:, 1:] if len(g) else np.zeros((0, 4))
                p = p[p[:, 0] == c] if len(p) else np.zeros((0, 6))
                n_gt += len(g)
                if len(p) == 0: continue
                order = np.argsort(-p[:, 1])
                p = p[order]
                matched = np.zeros(len(g), dtype=bool)
                ious = iou_matrix(p[:, 2:], g) if len(g) else np.zeros((len(p), 0))
                for i in range(len(p)):
                    scores.append(p[i, 1])
                    if len(g) == 0:
                        tps.append(0)
                        continue
                    j = int(np.argmax(ious[i]))
                    if ious[i, j] >= thr and not matched[j]:
                        matched[j] = True
                        tps.append(1)
                    else:
                        tps.append(0)
            if n_gt == 0: continue
            if not scores:
                cls_ap.append(0.0)
                continue
            order = np.argsort(-np.array(scores))
            tp = np.array(tps)[order]
            ctp = np.cumsum(tp)
            cfp = np.cumsum(1 - tp)
            rec = ctp / n_gt
            prec = ctp / np.maximum(ctp + cfp, 1e-9)
            cls_ap.append(voc_ap(rec, prec))
        aps.append(np.mean(cls_ap) if cls_ap else 0.0)
    return float(np.mean(aps))

def main():
    rng = np.random.default_rng(SEED)
    gt = load_ground_truth(GT_DIR)
    pa = load_predictions(PRED_A)
    pb = load_predictions(PRED_B)

    ids = sorted(list(set(gt.keys()) | set(pa.keys()) | set(pb.keys())))
    n = len(ids)
    print(f"评估集图像数: {n}")

    for label, thrs in [("mAP@0.5", IOU_50), ("mAP@0.5:0.95", IOU_COCO)]:
        base_a = compute_map(ids, gt, pa, thrs, N_CLASSES)
        base_b = compute_map(ids, gt, pb, thrs, N_CLASSES)

        deltas = []
        for b in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)
            sample = [ids[i] for i in idx]
            da = compute_map(sample, gt, pa, thrs, N_CLASSES)
            db = compute_map(sample, gt, pb, thrs, N_CLASSES)
            deltas.append(da - db)
            if (b + 1) % 100 == 0:
                print(f"  {label}: {b + 1}/{N_BOOTSTRAP}", end="\r")

        deltas = np.array(deltas)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())

        print(f"\n=== {label} ===")
        print(f"{NAME_A}: {base_a:.3f}   {NAME_B}: {base_b:.3f}")
        print(f"Delta = {base_a - base_b:+.3f}   95% CI: [{lo:+.3f}, {hi:+.3f}]   p ≈ {p:.3f}")
        if lo <= 0 <= hi:
            print("→ 区间包含 0：两模型差异在统计上不可区分（可写成 'comparable'）")
        else:
            print("→ 区间不含 0：差异在评估集抽样不确定性之外")

if __name__ == "__main__":
    main()