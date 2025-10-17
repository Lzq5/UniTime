import argparse
import json
import numpy as np
import os
import copy

def to_window_list_pred_vr(pred):
    windows = np.array(list(filter(lambda x: x != -100, pred)))
    window_list = windows.astype(str).tolist()
    window_list = [float(num) for num in window_list]
    if window_list == []:
        return [-1]
    return window_list

import torch
from nncore.ops import temporal_area, temporal_intersection
def compute_iou_multi(pred, span):
    pred_tensor = torch.Tensor(pred)
    span_tensor = torch.Tensor(span)
    pred_area = temporal_area(pred_tensor).sum()
    span_area = temporal_area(span_tensor).sum()
    inter = temporal_intersection(pred_tensor, span_tensor).sum()
    iou = (inter / (pred_area + span_area - inter)).unsqueeze(0)

    iou = torch.where(iou.isfinite(), iou, 0)
    return iou

def evaluate_performance(
        predictions, thresholds, topK, per_instance=False
):
    """Evalutes the performances."""
    results = [[[] for _ in topK] for _ in thresholds]
    average_IoU = []
    num_instances = 0

    for pred_datum in predictions:
        overlap = compute_iou_multi(pred_datum["pred_relevant_windows"],pred_datum["relevant_windows"]).unsqueeze(0).numpy()
        average_IoU.append(overlap[0])

        for tt, threshold in enumerate(thresholds):
            for rr, KK in enumerate(topK):
                results[tt][rr].append((overlap > threshold)[:KK].any())
        num_instances += 1
    mean_results = np.array(results).mean(axis=-1)
    mIoU = np.mean(average_IoU)
    print(f"Evaluated: {num_instances} instances")

    if per_instance:
        per_instance_results = {
            "overlap": overlap,
            "average_IoU": average_IoU,
            "results": results,
        }
        return mean_results, mIoU, per_instance_results
    else:
        return mean_results, mIoU

def get_metrics(results, mIoU, thresholds, topK):
    result_dict = {}
    results *= 100
    mIoU *= 100
    for ii in range(len(topK)):
        for jj in range(len(thresholds)):
            key = f"Rank@{topK[ii]}\nmIoU@{thresholds[jj]}"
            result_dict[key] = f"{results[jj][ii]:.02f}"

    result_dict["mIoU"] = f"{mIoU:.02f}"
    return result_dict

def save_json(content, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        f.write(json.dumps(content))


def main():
    parser = argparse.ArgumentParser(description='Run Qwen2 - VL inference on Metrics')

    parser.add_argument('--res', type=str, default=None, help='path to results')
    args = parser.parse_args()

    dataset_name_list = ['charades', 'qvhighlights', 'tacos', 'anet', 'ego4d']
    detected_dataset = None
    for dataset in dataset_name_list:
        if dataset in args.res:
            detected_dataset = dataset
            break
    if detected_dataset:
        print(f"Detected dataset: {detected_dataset}")
    else:
        print("No dataset name detected in the data path.")
    
    # [ToModify] path_to_test_data for each benchmark
    gt_paths = {
        "charades":"./datasets/charades/test_all.json",
        "ego4d":"./datasets/ego4d/val_all.json",
        "tacos":"./datasets/tacos/test_all.json",
        "anet":"./datasets/anet/test_all.json",
        "qvhighlights":"./datasets/qvhighlights/val_vc.json",
    }
    
    thresholds_dict ={
        "charades": [0.5, 0.7, 0.3],
        "qvhighlights": [0.5, 0.7],
        "ego4d": [0.3, 0.5, 0.7],
        "tacos": [0.3, 0.5, 0.7],    
        "anet": [0.5, 0.7, 0.3],
    }

    results_data = json.load(open(args.res))

    
    gt_path = gt_paths[detected_dataset]
    gt_data = json.load(open(gt_path))
    gt_label = {label["qid"]: label for label in gt_data}

    results_interpreted_gt = [
        {
            "qid": int(res['qid']),
            "pred_relevant_windows": res["pred_relevant_windows"] if res["pred_relevant_windows"] != [] else [[-1, -1]],
            "relevant_windows": gt_label[int(res['qid'])]["annos"][0]["window"],
        }
        for res in results_data
    ]

    thresholds = thresholds_dict[detected_dataset]
    topK = [1]
    results_gt, mIoU_gt = evaluate_performance(
        results_interpreted_gt, thresholds, topK
    )
    
    metrics = get_metrics(results_gt, mIoU_gt, thresholds, topK)
    print(metrics)


if __name__ == "__main__":
    main()