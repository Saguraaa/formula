import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path)
    p.add_argument('--data', type=Path, default=ROOT / 'datasets_oriented/yolo/data.yaml')
    p.add_argument('--weights', type=Path, default=ROOT / 'weights/yolo11n.pt')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='0')
    p.add_argument('--name', default='yolo11n-baseline')
    for parameter in ('degrees', 'translate', 'scale', 'shear', 'perspective'):
        p.add_argument(f'--{parameter}', type=float, default=0.0)
    p.add_argument('--swanlab-mode', choices=['online', 'offline', 'disabled'], default='online')
    config = p.parse_known_args()[0].config
    if config:
        import yaml

        options = yaml.safe_load(config.read_text(encoding='utf-8-sig'))
        if not isinstance(options, dict) or set(options) - {
            'data', 'weights', 'epochs', 'batch', 'imgsz', 'name',
            'degrees', 'translate', 'scale', 'shear', 'perspective',
        }:
            p.error('invalid YOLO config fields')
        for key in ('data', 'weights'):
            if key in options:
                options[key] = ROOT / options[key]
        p.set_defaults(**options)
    args = p.parse_args()
    from ultralytics import YOLO
    import swanlab
    from swanlab.integration.ultralytics import add_swanlab_callback
    model = YOLO(str(args.weights))
    add_swanlab_callback(model, project='ModelAdjust', experiment_name=args.name, mode=args.swanlab_mode)
    try:
        model.train(data=str(args.data), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                    device=args.device, workers=0, seed=42, project=str(ROOT / 'runs'), name=args.name,
                    fliplr=0, flipud=0, mosaic=0, mixup=0,
                    degrees=args.degrees, translate=args.translate, scale=args.scale,
                    shear=args.shear, perspective=args.perspective,
                    hsv_h=0, hsv_s=0, hsv_v=0, plots=True)
    finally:
        swanlab.finish()


if __name__ == '__main__': main()
