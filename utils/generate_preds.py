# 此脚本是用于第一篇论文一审时添加Bootsrap参数用的，这个脚本是第一步，real_bootstrap.py脚本是第二步，生成置信区间必须两个脚本结合着用，顺序不能打乱
from ultralytics import YOLO

# 1. 加载你训练好的最佳权重
# 路径已经帮你填好了，直接用
weight_path = '/home/featurize/work/yolov11-M4+Hyper+head/runs/train/exp3(no/weights/best.pt'
model = YOLO(weight_path)

print("⏳ 开始加载模型并进行验证集推理...")

# 2. 运行验证模式，并强制保存 TXT 预测结果和置信度
# 【注意】请把 data= 后面的路径替换成你真实的 yaml 配置文件路径
metrics = model.val(
    data='/home/featurize/work/yolov13-main/tranin_val1.yaml',  # <--- 修改这里：填入你的数据集 yaml 文件路径
    imgsz=640,          # 保持和你训练时一样的 640 尺寸
    save_txt=True,      # 关键参数：让模型把预测框保存成 txt
    save_conf=True,     # 关键参数：把置信度也保存在 txt 里
    split='val'         # 确保跑的是验证集
)

print("✅ 推理完成！")
print("请去 runs/detect/val/labels (或者类似的最新 exp 文件夹) 检查是否生成了大量的 txt 文件。")