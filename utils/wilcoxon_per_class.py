"""
    Ultralytics 的 model.val() 本来就会输出每个类别的 AP。
    Fish-BJ 有 21 个物种 → 每个模型有 21 个配对的 per-class AP 值。
    以「物种」为配对单位，对 ULFD-YOLO 与每个基线做 Wilcoxon 符号秩检验，
    得到 p 值。这正是 R4 原话里点名的检验之一，不需要重训练、不需要重采样，
    只要把各模型 val 的 per-class 结果抄进下面的字典即可。

同时输出 Hodges–Lehmann 中位差（配对差值的中位数）作为效应量。

依赖：scipy
用法：填好 PER_CLASS 后 python wilcoxon_per_class.py
"""

import numpy as np
from scipy.stats import wilcoxon

# 21 个 Fish-BJ 物种，顺序必须在所有模型之间一致
SPECIES = [
    'Pinktail Triggerfish',
    'Bigspotted Triggerfish',
    'Niger Triggerfish',
    'Speckled pavon',
    'tilapia buttikoferi',
    'white butterflyfish',
    'Rea Giant gourami',
    'Foxface rabbitfish',
    'Orangeshoulder Tang',
    'Tobies',
    'Giant gourami',
    'discus fish',
    'Bignose unicornfish',
    'Blood parrot',
    'Volitan Lionfish',
    'Striped Surgeonfish',
    'African mooney',
    'Myleus schomburgkii',
    'Spotted Scat',
    'Metynnis hypsauchen',
    'Archerfish'
]

# 从各模型的 val 输出里抄 per-class 值。
# 建议同时跑一遍 mAP@0.5 和一遍 mAP@0.5:0.95，各做一次检验。
METRIC = "mAP@0.5:0.95"   # 或 "mAP@0.5"

PER_CLASS = {
    "ULFD-YOLO": [0.815, 0.770, 0.713, 0.780, 0.688, 0.694, 0.811, 
        0.704, 0.665, 0.654, 0.809, 0.732, 0.727, 0.810, 
        0.801, 0.681, 0.696, 0.672, 0.654, 0.849, 0.719],   # 21 个值；论文 Table 3 已有 ULFD-YOLO 的 mAP@0.5:0.95 per-class
    "YOLOv11n":  [0.913, 0.868, 0.827, 0.915, 0.836, 0.782, 0.883,
        0.815, 0.791, 0.789, 0.844, 0.870, 0.848, 0.904,
        0.908, 0.871, 0.823, 0.755, 0.803, 0.944, 0.817]
    # "YOLOv8n":   [...],
    # "YOLOv5n":   [...],
    # "YOLOv3-tiny": [...],
    # "YOLOv13n":  [...],
    # "DETR":      [...],
}

OURS = "ULFD-YOLO"


def main():
    assert OURS in PER_CLASS, "请先填入 ULFD-YOLO 的 per-class 值"
    a = np.asarray(PER_CLASS[OURS], dtype=float)
    assert len(a) == len(SPECIES), f"{OURS} 应有 {len(SPECIES)} 个 per-class 值"

    print(f"检验指标: {METRIC}   配对单位: 物种 (n = {len(SPECIES)})")
    print(f"基准模型: {OURS}  (mean = {a.mean():.3f})")
    print("=" * 78)
    print(f"{'Baseline':<14}{'mean':>8}{'median Δ':>11}{'W':>8}{'p (two-sided)':>16}  结论")
    print("-" * 78)

    for name, vals in PER_CLASS.items():
        if name == OURS:
            continue
        b = np.asarray(vals, dtype=float)
        assert len(b) == len(SPECIES), f"{name} 应有 {len(SPECIES)} 个 per-class 值"
        d = a - b
        if np.allclose(d, 0):
            print(f"{name:<14}{b.mean():>8.3f}{0.0:>11.3f}{'-':>8}{'-':>16}  完全相同")
            continue
        stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        med = float(np.median(d))
        verdict = "差异不显著 (comparable)" if p >= 0.05 else ("我们更优" if med > 0 else "基线更优")
        print(f"{name:<14}{b.mean():>8.3f}{med:>+11.3f}{stat:>8.1f}{p:>16.4f}  {verdict}")

    print("=" * 78)
    print("（p >= 0.05）：")
    print('  "A Wilcoxon signed-rank test across the 21 Fish-BJ species found no significant')
    print('   difference between ULFD-YOLO and <baseline> under mAP@0.5:0.95 (p = 0.xxx),')
    print('   supporting the description of these detectors as comparable under the stricter')
    print('   localization criterion."')
    print()
    print("注意：n = 21，检验效力有限；p >= 0.05 只能说「未发现显著差异」，")



if __name__ == "__main__":
    main()
