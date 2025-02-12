import os
import argparse
import torch
import shutil
import yaml

from ultralytics import YOLO
from ultralytics import utils

from MoaiPipelineManager import Manager

utils.ONLINE = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="20250123", type=str)
    parser.add_argument("--subproject", default="sub", type=str)
    parser.add_argument("--task", default="obb", type=str)
    parser.add_argument("--version", default="v3", type=str)
    args = parser.parse_args()
    return args

def main(args):
    manager = Manager(**vars(args))

    hyp_yaml_path = manager.get_hyp_yaml_path()
    with open(hyp_yaml_path, "r") as f:
        hyp = yaml.safe_load(f)

    weight_path = manager.get_best_weight_path()
    print(weight_path)
    model = YOLO(weight_path)

    ARGS = {
        'project'     : f"{manager.location}/{manager.project}/{manager.subproject}/{manager.task}/{manager.version}",
        'name'        : "inference_result",
        'source'      : manager.get_test_dataset_path(),
        'imgsz'       : hyp['imgsz'],
        'device'      : DEVICE,
        'conf'        : 0.1,
        'save_txt'    : False,
        'save'        : False,
        'show_labels' : False,
        'show_conf'   : False,
        'show_boxes'  : False
    }

    inference_results = model.predict(**ARGS)

    save_path = f"{manager.location}/{manager.project}/{manager.subproject}/{manager.task}/{manager.version}/inference_result"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for result in inference_results:
        # Prepare filename and save path
        filename = os.path.basename(result.path)
        base, _ = os.path.splitext(filename)
        new_filename = base + '.txt'
        classes = result.names

        obb_results = []

        for i in range(len(result.obb)):
            cls_name = classes[int(result.obb.cls.cpu().numpy().tolist()[i])]
            conf = result.obb.conf.cpu().numpy().tolist()[i]
            xywhr = result.obb.xywhr.cpu().numpy().tolist()[i]  # shape = [N, 5]

            obb_results.append([cls_name, conf, xywhr])

        if len(obb_results) == 0:
            # Create empty file when no OBB is found
            open(os.path.join(save_path, new_filename), 'w').close()
            continue
        
        result_content = []
        # 여러 바운딩 박스를 순회하며 그리기
        for i in range(len(obb_results)):
            cls_name = obb_results[i][0]
            conf = obb_results[i][1]
            cx, cy, w, h, r = obb_results[i][2]

            # w, h가 음수/영인 경우 방어적 처리
            if w <= 0 or h <= 0:
                continue

            result_content.append([
                cls_name,
                cx, cy, w, h, r,
                conf
            ])

        with open(os.path.join(save_path, new_filename), "w") as f:
            for content in result_content:
                f.write(f"{content[0]} {content[1]} {content[2]} {content[3]} {content[4]} {content[5]} {content[6]}\n")


if __name__ == '__main__':
    args = get_args()
    main(args)
