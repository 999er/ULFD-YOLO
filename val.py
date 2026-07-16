from ultralytics import YOLO

# 1. 加载你训练好的最佳权重
model = YOLO('/home/featurize/work/yolov11-M4+Hyper+head/runs/train/exp/weights/bast.pt')

# 2. 运行验证模式
# 请确保 data 参数指向你真实的 yaml 文件
metrics = model.val(data='/home/featurize/work/yolov13-main/tranin_val.yaml', imgsz=640)