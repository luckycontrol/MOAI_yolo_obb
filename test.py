import os
import argparse
import torch
import shutil

from ultralytics import YOLO

from MoaiPipelineManager import Manager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    weight_path = manager.get_best_weight_path()
    hyp_yaml_path = manager.get_hyp_yaml_path()
    with open(hyp_yaml_path, "r") as f:
        hyp = yaml.safe_load(f)

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

    for result in inference_results:
        names = result.names
        classes = result.obb.cls.cpu().numpy().tolist()
        confs = result.obb.conf.cpu().numpy().tolist()
        
        # 검출 결과 가져오기
        xywhr = result.obb.xywhr  # shape = [N, 5]

        # Prepare filename and save path
        filename = os.path.basename(result.path)
        base, _ = os.path.splitext(filename)
        new_filename = base + '.txt'
        save_path = f"{manager.location}/{manager.project}/{manager.subproject}/{manager.task}/{manager.version}/inference_result"

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        if xywhr.shape[0] == 0:
            # Create empty file when no OBB is found
            open(os.path.join(save_path, new_filename), 'w').close()
            continue
        
        result_content = []
        # 여러 바운딩 박스를 순회하며 그리기
        for i in range(xywhr.shape[0] - 1):
            # 텐서 -> CPU -> numpy 변환
            cx, cy, w, h, r = xywhr[i].cpu().numpy().tolist()

            # w, h가 음수/영인 경우 방어적 처리
            if w <= 0 or h <= 0:
                continue

            result_content.append([
                names[classes[i]],
                cx, cy, w, h, r,
                confs[i]
            ])

        with open(save_path + "/" + new_filename, "w") as f:
            for content in result_content:
                f.write(f"{content[0]} {content[1]} {content[2]} {content[3]} {content[4]} {content[5]} {content[6]}\n")


if __name__ == '__main__':
    args = get_args()
    main(args)
