import argparse
import torch
from torch.nn.utils.rnn import pad_sequence
import re
import numpy as np
from models.qwen2_vl import Qwen2VLMRForConditionalGeneration, Qwen2VLMRProcessor
from qwen_vision_process import process_vision_info
from feature import feature
import json
import os
from tqdm import tqdm

PAD_IDX = -100

def extract_time(sentences):
    results = []
    for sentence in sentences:
        matches = re.findall(r"(\d+(\.\d+)?)", sentence)
        if matches:
            results.append(torch.tensor([float(match[0]) for match in matches]))
        else:
            results.append(torch.tensor([PAD_IDX]))
    results = pad_sequence(results,batch_first=True,padding_value=PAD_IDX)
    return results

def to_window_list_pred(pred):
    windows = np.array(list(filter(lambda x: x != PAD_IDX, pred)))
    if len(windows) == 0:
        return [[-1,-1]]
    if len(windows) % 2 != 0:
        windows = windows[:-1]
    window_list = windows.reshape(-1, 2).astype(str).tolist()
    window_list = [[float(num) for num in pair] for pair in window_list]
    if window_list == []:
        return [[-1,-1]]
    return window_list

def to_window_list_pred_vr(pred):
    windows = np.array(list(filter(lambda x: x != PAD_IDX, pred)))
    window_list = windows.astype(str).tolist()
    window_list = [float(num) for num in window_list]
    if window_list == []:
        return [-1]
    return window_list

def construct_messages_mr_fps(video_path, feature_path, fps, retrieval_segment, retrieval_mode, clip_length):
    """Constructs input message for the model based on the retrieval mode."""
    if retrieval_mode == 'mr_seg':
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"{video_path}", "fps": fps, "video_start": retrieval_segment[0], "video_end": retrieval_segment[1], 
                        "feature": f"{feature_path}", "num_clips": 1, "clip_length": clip_length},
                    {"type": "text", "text": f"This is a sequence interleaved with timestamps and frames. Your task is to identify the specific timestamp(s) when the given query appears."}
                ]
            },
        ]
    elif retrieval_mode == 'mr':
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"{video_path}", "fps": fps, "video_start": retrieval_segment[0], "video_end": retrieval_segment[1]},
                    {"type": "text", "text": f"This is a sequence interleaved with timestamps and frames. Your task is to identify the temporal window (start and end timestamps) when the given query appears."}
                ]
            },
        ]
    return message


def run_inference(model, processor, data, args, device):
    qid = data["qid"]
    vid = data["id"]
    annos = data["annos"]

    video_start = data.get("video_start", 0)
    video_end = data.get("video_end", data["duration"])
    temporal_windows = [anno["window"] for anno in annos]
    querys = [anno["query"] for anno in annos]

    retrieval_segment = [video_start, video_end]

    if video_end - video_start <= 256:
        retrieval_mode = "mr"
    else:
        retrieval_mode = "mr_seg"

    video_path = data.get("video_path", None)
    feature_path = data.get("feature_path", None)
    if feature_path is None and retrieval_mode == 'mr_seg':
        feature_path = feature(model, processor, video_path, feature_root=args.feat_folder)

    message = construct_messages_mr_fps(
        video_path, feature_path, args.fps, retrieval_segment, retrieval_mode, args.clip_length
    )
    messages = [message]

    # Process vision inputs with the processor
    image_inputs, video_inputs, all_timestamps_combine, feature_inputs, combine_t_list = process_vision_info(messages)

    if feature_inputs is None:
        all_timestamps_num = [[round((x + y)/2, 1) for x, y in zip(sublist[::2], sublist[1::2])] for sublist in all_timestamps_combine]
    else:
        all_timestamps_num = all_timestamps_combine
    all_timestamps = [[f"timestamp: {all_t} seconds; feature: " for all_t in sublist] for sublist in all_timestamps_num]

    results = []
    for query, temporal_window in zip(querys, temporal_windows):
        message_for_query = message.copy()
        message_for_query.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Query:{query}\nAnswer: "}
                ]
            }
        )
        text = processor.apply_chat_template(
            message_for_query, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            features=feature_inputs,
            timestamps=all_timestamps,
            combine_t_list=combine_t_list,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        if feature_inputs is not None:
            feature_inputs = torch.cat(
                [feature_inputs[i].reshape(-1, feature_inputs[i].shape[3]) for i in range(len(feature_inputs))],
                dim=0
            )
        else:
            feature_inputs = None
        if combine_t_list is not None:
            combine_t_list = [torch.tensor(i) for i in combine_t_list]
        else:
            combine_t_list = None
        if 'pixel_values_videos' in inputs:
            pixel_values_videos = inputs['pixel_values_videos']
        else:
            pixel_values_videos = None 

        model_inputs = dict(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=inputs['video_grid_thw'],
                feature_inputs=feature_inputs,
                combine_t_list=combine_t_list
            )
        
        gen_kwargs = {'max_new_tokens': 128, 'temperature': 0.0, 'top_p': 1.0, 'num_beams': 1, 'do_sample': False}

        # Generate predictions
        generated_ids = model.generate(
            **model_inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            do_sample=True if gen_kwargs["temperature"] > 0 else False,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
            use_cache=True,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


        predictions = extract_time(output_text).numpy()
        if retrieval_mode =='mr':
            data_dict = {
                "qid": qid,
                "id": vid,
                "annos": [
                    {
                        "query": query,
                        "window": temporal_window,
                    }
                ],
                "duration": data['duration'],
                "video_path": video_path,
                "feature_path": feature_path,
                "pred_relevant_windows": to_window_list_pred(predictions[0]),
                "pred_relevant_windows_mr_seg": data.get("pred_relevant_windows_mr_seg", None)
            }
            results.append(data_dict)
        else:
            pred_relevant_windows_mr_seg = data.get("pred_relevant_windows_mr_seg", None)
            if pred_relevant_windows_mr_seg is None:
                pred_relevant_windows_mr_seg = []
            pred_relevant_windows_mr_seg.append(to_window_list_pred_vr(predictions[0]))
            data["pred_relevant_windows_mr_seg"] = pred_relevant_windows_mr_seg

            sampled_timestamps = torch.tensor(all_timestamps_num[0])
            predictions_seg = to_window_list_pred_vr(predictions[0])
            video_start = predictions_seg[0]
            video_end = predictions_seg[-1]
            duration = data["duration"]
            try:
                predict_end_index = sampled_timestamps.index(video_end)
                if predict_end_index == len(sampled_timestamps) - 1:
                    video_end = duration
                else:
                    video_end = sampled_timestamps[predict_end_index + 1]
            except:
                clip_length = sampled_timestamps[1] - sampled_timestamps[0]
                video_end = video_start + clip_length
            video_end = min(video_end, duration)
            video_start = max(0, video_start)

            data["video_start"] = video_start
            data["video_end"] = video_end
            return run_inference(model, processor, data, args, device)

    return results


def main():
    parser = argparse.ArgumentParser(description='Run Qwen2-VL inference on Temporal Grounding')
    parser.add_argument('--model_local_path', type=str, default='/raid/haoningwu/zeqianli/models/Qwen2-VL-7B-Instruct',
                        help='Path to the local model')
    parser.add_argument('--output_dir', type=str, default='debug')
    parser.add_argument('--model_finetune_path', type=str, default=None, help='Path to the finetune model')
    parser.add_argument('--data_path', type=str, default='./datasets/cgbench/val_vc.json', help='Path to the data JSON file')
    parser.add_argument('--feat_folder', type=str, default='./tmp_feature')
    parser.add_argument('--fps', type=int, default=2)
    parser.add_argument('--clip_length', default=32, type=int)
    args = parser.parse_args()

    # Load model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_finetune_path:
        model = Qwen2VLMRForConditionalGeneration.from_pretrained(args.model_finetune_path, torch_dtype=torch.bfloat16, device_map={"": device}, attn_implementation="flash_attention_2")
    else:
        model = Qwen2VLMRForConditionalGeneration.from_pretrained(args.model_local_path, torch_dtype=torch.bfloat16, device_map={"": device}, attn_implementation="flash_attention_2")
    model.eval()
    processor = Qwen2VLMRProcessor.from_pretrained(args.model_local_path)

    # Load dataset
    dataset = json.load(open(args.data_path, "r"))

    # Run inference
    all_results = []
    for data in tqdm(dataset[:10], desc="Processing"):
        results = run_inference(model, processor, data, args, device)
        all_results.extend(results)
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()