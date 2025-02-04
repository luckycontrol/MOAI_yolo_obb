import argparse
import yaml

from ultralytics import YOLO
from MoaiPipelineManager import Manager

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", default="20250115", type=str)
    parser.add_argument("--subproject", default="test_sub", type=str)
    parser.add_argument("--task", default="test_obb", type=str)
    parser.add_argument("--version", default="v1", type=str)

    args = parser.parse_args()

    return args

def main(args):
    manager = Manager(**vars(args))

    arg_path = f"{manager.get_training_result_folder_path()}/args.yaml"
    with open(arg_path, "r") as f:
        arg = yaml.safe_load(f)

    weight_path = manager.get_best_weight_path()
    model = YOLO(weight_path)

    ARGS = {
        'format'    : 'onnx',
        'opset'     : 16,
        'imgsz'     : arg['imgsz'],
    }

    model.export(**ARGS)

if __name__ == "__main__":
    args = get_args()
    main(args)