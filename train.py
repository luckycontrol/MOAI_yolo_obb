import os
import argparse
import torch
import shutil
import yaml

from ultralytics import YOLO

from MoaiPipelineManager import Manager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", default="20250115", type=str)
    parser.add_argument("--subproject", default="test_sub", type=str)
    parser.add_argument("--task", default="test_obb", type=str)
    parser.add_argument("--version", default="v6", type=str)

    args = parser.parse_args()

    return args

def main(args):
    manager = Manager(**vars(args))

    data_path = manager.get_data_yaml_path()
    with open(data_path, "r+") as f:
        data = yaml.safe_load(f)
        data["train"] = f"{manager.get_train_dataset_path()}/train/images"
        data["val"] = f"{manager.get_train_dataset_path()}/valid/images"

    os.remove(data_path)

    with open(data_path, "w") as f:
        yaml.dump(data, f)

    hyp = manager.get_hyp_yaml()
    
    weights = hyp['weights']
    model = YOLO(f"{os.getcwd()}/weights/yolo11{weights}.pt")

    ARGS = {}
    ARGS.update(hyp)

    if 'weights' in ARGS:
        ARGS.pop('weights')  # 이미 model = YOLO(...)에서 처리했으므로 제거
    if 'batch_size' in ARGS:
        # batch_size -> batch로 변경
        ARGS['batch'] = ARGS.pop('batch_size')

    ARGS['project'] = f"{manager.location}/{manager.project}/{manager.subproject}/{manager.task}/{manager.version}"
    ARGS['name'] = 'training_result'
    ARGS['data'] = manager.get_data_yaml_path()
    ARGS['lr0'] = 0.0005
    ARGS['lrf'] = 0.0005
    ARGS['optimizer'] = 'AdamW'
    ARGS['device'] = DEVICE

    if 'train_ratio' in ARGS:
        ARGS.pop('train_ratio')
    
    if 'use_valid' in ARGS:
        ARGS.pop('use_valid')

    if 'valid_ratio' in ARGS:
        ARGS.pop('valid_ratio')

    model.train(**ARGS)

    # weights 폴더를 training_result 폴더로 이동
    # training_result_path = manager.get_training_result_folder_path()
    # weights_folder_path = manager.get_weight_folder_path()
    # shutil.move(f"{training_result_path}/weights", weights_folder_path)

if __name__ == '__main__':
    args = get_args()

    main(args)