from ultralytics import YOLO
from ultralytics import settings

def main():
    settings.update({
        'runs_dir': 'runs',
        'tensorboard': False,
    })
    model = YOLO('yolov5s.pt')
    model.train(
        # hsv_h=0.015,
        # hsv_s=0.7,
        # hsv_v=0.4,
        data=conf_path,
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        degrees=0,
        translate=0.3,
        scale=0.4,
        mosaic=0.5,
        mixup=0.3,
        erasing=0.4,
        crop_fraction=1.0,
        project='runs/detect',
        device = 'cuda:0',
        # resume=True,
        # amp=True
    )


if __name__ == '__main__':
    img_size = 32*16
    epochs = 1000
    conf_path = 'exported/project-51-at-2024-08-03-18-42-40c32cde/conf.yaml'
    main()
