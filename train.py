import os
import argparse
import torch
import shutil

from ultralytics import YOLO

from MoaiPipelineManager import Manager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def on_fit_epoch_end(trainer):
    numeric_metrics = {}
    for k, v in trainer.metrics.items():
        # float로 변환 가능한 값만 따로 모음
        if isinstance(v, (float, int)):
            numeric_metrics[k] = v
    # 이제 numeric_metrics 딕셔너리만 TensorBoard로 로깅
    _log_scalars(numeric_metrics, trainer.epoch + 1)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", default="20250115", type=str)
    parser.add_argument("--subproject", default="test_sub", type=str)
    parser.add_argument("--task", default="test_obb", type=str)
    parser.add_argument("--version", default="v3", type=str)

    args = parser.parse_args()

    return args

def main(args):
    manager = Manager(**vars(args))

    hyp = manager.get_hyp_yaml()
    
    weights = hyp['weights']
    model = YOLO(f"{os.getcwd()}/weights/yolo11{weights}.pt")

    ARGS = {k: v for (k, v) in hyp.items()}

    if 'weights' in ARGS:
        ARGS.pop('weights')  # 이미 model = YOLO(...)에서 처리했으므로 제거
    if 'batch_size' in ARGS:
        # batch_size -> batch로 변경
        ARGS.pop('batch_size')  

    ARGS = {
        'project'   : f"{manager.location}/{manager.project}/{manager.subproject}/{manager.task}/{manager.version}",
        'name'      : 'training_result',
        'data'      : manager.get_data_yaml_path(),
        'lr0'       : 0.0005,
        'lrf'       : 0.0005,
        'optimizer' : 'AdamW',
        'epochs'    : hyp['epochs'],
        'batch'     : hyp['batch_size'],
        'patience'  : hyp['epochs'],
        'device'    : DEVICE,
        'resume'    : hyp['resume'],
    }
        

    model.train(**ARGS)

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    # weights 폴더를 training_result 폴더로 이동
    training_result_path = manager.get_training_result_folder_path()
    weights_folder_path = manager.get_weight_folder_path()
    shutil.move(f"{training_result_path}/weights", weights_folder_path)

if __name__ == '__main__':
    args = get_args()

    main(args)