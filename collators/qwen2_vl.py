import re
from typing import Dict, Optional

import torch
from transformers import PreTrainedTokenizer, AutoProcessor, AutoConfig

from . import register_collator
from .base import BaseDataCollator
from .qwen_vision_process import process_vision_info
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

PAD_IDX = -100
DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"

def find_closest_timestamps(sample_timestamps, gt_window):
    contains_value = (sample_timestamps[0] <= gt_window[1] and sample_timestamps[-1] >= gt_window[0])

    candidates_start = [x for x in sample_timestamps if x <= gt_window[0]]
    if candidates_start:
        closest_start = max(candidates_start)
    else:
        closest_start = sample_timestamps[0]
    start_idx = sample_timestamps.index(closest_start)

    candidates_end = [x for x in sample_timestamps if x >= gt_window[1]]
    if candidates_end:
        closest_end = min(candidates_end)
    else:
        closest_end = sample_timestamps[-1]
    end_idx = sample_timestamps.index(closest_end)

    return contains_value, closest_start, closest_end, start_idx, end_idx

def find_segments(sample_timestamps, gt_window):

    candidates_start = [x for x in sample_timestamps if x <= gt_window[0]]
    if candidates_start:
        closest_start = max(candidates_start)
    else:
        closest_start = sample_timestamps[0]
    start_idx = sample_timestamps.index(closest_start)

    candidates_end = [x for x in sample_timestamps if x <= gt_window[1]]
    if candidates_end:
        closest_end = max(candidates_end)
    else:
        closest_end = sample_timestamps[0]
    end_idx = sample_timestamps.index(closest_end)

    return start_idx, end_idx

@register_collator("qwen2-vl")
class Qwen2VLDataCollator(BaseDataCollator):
    def __init__(
        self,
        config: Optional[AutoConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        processor: Optional[AutoProcessor] = None,
        mask_question_tokens: bool = True
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.mask_question_tokens = mask_question_tokens
        self.default_instances = None

    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def __call__(self, instances) -> Dict[str, torch.Tensor]:
        if self.default_instances == None:
            self.default_instances = instances
        split = instances[0]["split"]
        mode = instances[0]["mode"]
        temporal_window = [instance["temporal_window"] for instance in instances]
        messages = [instance["message"] for instance in instances]

        image_inputs, video_inputs, all_timestamps_combine, feature_inputs, combine_t_list = process_vision_info(messages)

        if feature_inputs is None:
            all_timestamps_num = [[round((x + y)/2, 1) for x, y in zip(sublist[::2], sublist[1::2])] for sublist in all_timestamps_combine]
        else:
            all_timestamps_num = all_timestamps_combine
        all_timestamps_origin = all_timestamps_num
        all_timestamps = [[f"timestamp: {all_t} seconds; feature: " for all_t in sublist] for sublist in all_timestamps_num]
        
        if mode == 'mr_seg':
            for msg, all_t, all_t_o, windows in zip(messages, all_timestamps_num, all_timestamps_origin, temporal_window):
                num_query = 0
                for t_w in windows:
                    sub_evaluate_labels = []
                    for t_w_i in t_w:
                        segment_start_idx, segment_end_idx = find_segments(all_t_o, t_w_i)
                        sub_evaluate_labels.extend([all_t[iii] for iii in range(segment_start_idx, segment_end_idx + 1)])
                    interval_text = ", ".join([f"{s} seconds" for s in sub_evaluate_labels])
                    interval_text = interval_text + "."
                    msg.insert(2*num_query+2, {"role": "assistant", "content": [{"type": "text", "text": f"{interval_text}"}]})
                    num_query += 1
        else:
            for msg, all_t, all_t_o, windows in zip(messages, all_timestamps_num, all_timestamps_origin, temporal_window):
                num_query = 0
                for t_w in windows:
                    sub_evaluate_labels = []
                    for t_w_i in t_w:
                        is_inside, s_t_i, e_t_i, s_t_idx, e_t_idx = find_closest_timestamps(all_t_o,t_w_i)
                        sub_evaluate_labels.append([all_t[s_t_idx], all_t[e_t_idx]])
                    interval_text = ", ".join([f"from {s} seconds to {e} seconds" for s, e in sub_evaluate_labels])
                    interval_text = interval_text[0].upper() + interval_text[1:] + "."
                    msg.insert(2*num_query+2, {"role": "assistant", "content": [{"type": "text", "text": f"{interval_text}"}]})
                    num_query += 1

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in messages
        ]

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            features=feature_inputs,
            timestamps=all_timestamps,
            combine_t_list=combine_t_list,
            padding=True,
            return_tensors="pt",
        ) 

        input_ids = inputs['input_ids']

        im_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        labels = input_ids.clone()
        seq_len = labels.shape[1]
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril(diagonal=0).unsqueeze(0).unsqueeze(0).repeat(labels.shape[0],1,1,1)
        try:
            attention_mask_multiqa = []
            for input_ids_bs, label_bs, mask in zip(input_ids, labels, causal_mask):
                indices = list(filter(lambda i: input_ids_bs[i] == im_start_token_id, range(len(input_ids_bs))))
                prompt_indices = [0, indices[2]]
                num_qa = (len(indices) - 2) // 2
                question_indices = [(indices[2 + i * 2], indices[2 + i * 2 + 1]) for i in range(num_qa)]
                answer_indices = [(indices[2 + i * 2 + 1], indices[2 + i * 2 + 2]) for i in range(num_qa - 1)]
                answer_indices.append([indices[-1], len(input_ids_bs)])
                response_mask = torch.zeros(len(input_ids_bs), dtype=torch.bool)
                for start, end in answer_indices:
                    response_mask[start:end] = True
                label_bs.masked_fill_(~response_mask, self.IGNORE_TOKEN_ID)

                pad_mask = input_ids_bs.ne(self.PAD_TOKEN_ID)
                pad_mask = pad_mask.unsqueeze(0).expand(seq_len, seq_len).unsqueeze(0)
                mask = mask * pad_mask

                for i in range(1, num_qa):
                    mask[0, question_indices[i][0]:answer_indices[i][1], prompt_indices[1]:answer_indices[i-1][1]] =  False
                
                attention_mask_multiqa.append(mask)
            attention_mask_multiqa = torch.stack(attention_mask_multiqa, dim=0)
        except:
            print(instances)
        min_dtype = torch.finfo(torch.bfloat16).min
        attention_mask_multiqa = torch.where(attention_mask_multiqa == 1, torch.tensor(0.0), min_dtype)
        attention_mask_multiqa = AttentionMaskConverter._unmask_unattended(attention_mask_multiqa, min_dtype)

        if 'attention_mask' in inputs:
            attention_mask = inputs['attention_mask']
        else:
            attention_mask = input_ids.ne(self.PAD_TOKEN_ID)

        if split == 'train':
            multi_qa = True
            attention_mask_multiqa = attention_mask_multiqa
        else:
            multi_qa =  False
            attention_mask_multiqa = None

        if 'pixel_values_videos' in inputs:
            pixel_values_videos = inputs['pixel_values_videos']
        else:
            pixel_values_videos = None 
        if 'video_grid_thw' in inputs:
            video_grid_thw = inputs['video_grid_thw']
        else:
            video_grid_thw = None

        if feature_inputs is not None:
            feature_inputs = torch.cat(
                [feature_inputs[i].reshape(-1, feature_inputs[i].shape[3]) for i in range(len(feature_inputs))],
                dim=0
            )
        else:
            feature_inputs = None

        if combine_t_list is not None:
            combine_t_list = [torch.tensor(i) for i in combine_t_list]
            combine_t_list = pad_sequence(combine_t_list, batch_first=True,padding_value=PAD_IDX)

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            labels=labels,
            feature_inputs=feature_inputs,
            multi_qa=multi_qa,
            attention_mask_multiqa=attention_mask_multiqa,
            combine_t_list=combine_t_list,
        )