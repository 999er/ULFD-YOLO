import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'/home/featurize/work/yolov11-M4+Hyper+head/ultralytics/cfg/models/11/yolov11-M4+Hyper.yaml')
    model.load('/home/featurize/work/yolov13-main/yolo11n.pt') # 加载预训练权重
    model.train(data=r'/home/featurize/work/yolov13-main/tranin_val.yaml',
                imgsz=640,
                epochs=300,
                batch=512,
                workers=2,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='/home/featurize/work/yolov11-M4+Hyper+head/runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                seed=42,             # <--- 在这里显式设置你的随机种子（比如 42，或者保持默认的 0 也可以）
                deterministic=True,  # <--- 加上这个开启确定性算法，确保完美复现，一般情况下把这两行注释掉
                )
