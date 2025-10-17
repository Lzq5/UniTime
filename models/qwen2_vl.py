from typing import Any, Dict, List, Optional, Tuple, Union
import torch 
from transformers.utils import logging

logger = logging.get_logger(__name__)

from transformers import Qwen2VLForConditionalGeneration
from torch import nn 
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast, Qwen2VisionTransformerPretrainedModel, Qwen2VLModel
import torch.nn.functional as F 
from transformers.cache_utils import StaticCache
from torch.nn import CrossEntropyLoss
from collators.qwen2_vl import PAD_IDX
import numpy as np
from transformers.utils import is_torchdynamo_compiling

class Qwen2VLMRForConditionalGeneration(Qwen2VLForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)

        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(
            config.vision_config, attn_implementation="flash_attention_2"
        )
        self.model = Qwen2VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.padding_side = "left"  # set it to left by default, user can use setter to change padding_sides
        self.rope_deltas = None

        # Initialize weights and apply final processing
        self.post_init()

    def encode_video_chunk(self, pixel_values_videos, video_grid_thw, combine_t_list):
        video_embeds = []
        start = 0
        video_batch_size = 8
        video_index = 0
        for combine_t in combine_t_list:
            t = len(combine_t)
            thw = video_grid_thw[video_index]
            for t_start in range(0, t, video_batch_size):
                video_grid_thw_batch = torch.tensor([[video_batch_size,thw[1],thw[2]]]).to(pixel_values_videos.device)
                t_end = min(t_start + video_batch_size, t)
                if t_end - t_start < video_batch_size:
                    seq_length = t_end - t_start
                    pad_length = video_batch_size - seq_length
                    end = start + seq_length * thw[1] * thw[2]
                    pixel_values_videos_batch = pixel_values_videos[start:end]
                    padded_pixel_values_videos_batch = torch.cat([pixel_values_videos_batch, pixel_values_videos_batch[-int(thw[1] * thw[2]):].repeat(pad_length,1)], dim=0)
                    video_embeds_chunk = self.visual(padded_pixel_values_videos_batch, grid_thw=video_grid_thw_batch)
                    video_embeds.append(video_embeds_chunk[:int(seq_length * thw[1] * thw[2] / 4)])
                else:
                    end = start + video_grid_thw_batch[0].prod()
                    pixel_values_videos_batch = pixel_values_videos[start:end]
                    video_embeds_chunk = self.visual(pixel_values_videos_batch, grid_thw=video_grid_thw_batch)
                    video_embeds.append(video_embeds_chunk)
                start = end
                del video_embeds_chunk
            video_index += t
        video_embeds = torch.cat(video_embeds).to(pixel_values_videos.device)
        return video_embeds
    
    def get_rope_index_multiqa(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []

        im_start_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")

        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i].to(input_ids.device) == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                indices = list(filter(lambda i: input_ids[i] == im_start_token_id, range(len(input_ids))))
                prompt_indices = [0, indices[2]]
                num_qa = (len(indices) - 2) // 2
                question_indices = [(indices[2 + i * 2], indices[2 + i * 2 + 1]) for i in range(num_qa)]
                answer_indices = [(indices[2 + i * 2 + 1], indices[2 + i * 2 + 2]) for i in range(num_qa - 1)]
                answer_indices.append([indices[-1], len(input_ids)])

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = prompt_indices[1] - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                st_idx_multiqa = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0

                for n_qa in range(num_qa):
                    text_len = answer_indices[n_qa][1] - question_indices[n_qa][0]
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx_multiqa)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        feature_inputs=None,
        multi_qa=False,
        attention_mask_multiqa=None,
        combine_t_list=None,
        **kwargs,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            if pixel_values_videos is not None or feature_inputs is not None:
                if feature_inputs is not None:
                    video_embeds = feature_inputs
                else:
                    pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                    video_embeds = self.encode_video_chunk(pixel_values_videos, video_grid_thw, combine_t_list).to(inputs_embeds.device)
                    video_embeds.requires_grad=True

                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                reshape_video_grid_thw = video_grid_thw
                
                if multi_qa:
                    position_ids, rope_deltas = self.get_rope_index_multiqa(
                        input_ids, image_grid_thw, reshape_video_grid_thw, attention_mask
                    )
                else:
                    position_ids, rope_deltas = self.get_rope_index(
                        input_ids, image_grid_thw, reshape_video_grid_thw, attention_mask
                    )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            
        
        if attention_mask_multiqa is not None:
            attention_mask = attention_mask_multiqa.to(inputs_embeds.device)
    
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
            elif (
                inputs_embeds is not None  # Exception 1
                or (is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1])  # Exception 3
            ):
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        feature_inputs = kwargs.get("feature_inputs", None)
        multi_qa = kwargs.get("multi_qa", False)
        attention_mask_multiqa = kwargs.get("attention_mask_multiqa", None)
        combine_t_list = kwargs.get("combine_t_list", None)

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None
            feature_inputs = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "cache_position": cache_position,
                "feature_inputs": feature_inputs,
                "multi_qa": multi_qa,
                "attention_mask_multiqa": attention_mask_multiqa,
                "combine_t_list": combine_t_list,
            }
        )
        return model_inputs
    
    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(input_ids)

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos" and dict_to_expand[key] is not None:
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    if not isinstance(dict_to_expand[key], list):
                        raise TypeError(
                            f"Expected value for key '{key}' to be a list, but got {type(dict_to_expand[key])} instead."
                        )
                    tensor = torch.tensor(dict_to_expand[key])
                    lengths = list(video_nums)
                    tensor = _repeat_interleave_samples(tensor, lengths=lengths, repeat_times=expand_size)
                    dict_to_expand[key] = tensor.tolist()
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        # input_ids is required for expanding visual inputs
        # If input_ids is unavailable, visual inputs will not be used; therefore, there is no need to expand visual inputs.
        if input_ids is not None and input_ids.numel() != 0:
            model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.image_utils import ImageInput, VideoInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessorKwargs, Qwen2VLProcessor

class Qwen2VLMRProcessor(Qwen2VLProcessor):
    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        features=None,
        timestamps=None,
        combine_t_list=None,
        **kwargs: Unpack[Qwen2VLProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Qwen2VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if features is None:
            if videos is not None:
                videos_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["videos_kwargs"])
                video_grid_thw = videos_inputs["video_grid_thw"]
                video_grid_thw = [[torch.tensor(1).to(thw.device),thw[1],thw[2]] for thw in video_grid_thw for i in range(thw[0])]
                videos_inputs["video_grid_thw"] = torch.tensor(video_grid_thw)
            else:
                videos_inputs = {}
                video_grid_thw = None
        else:
            videos_inputs = {}
            video_grid_thw = [torch.tensor([features[i].shape[0], features[i].shape[1] * 2, features[i].shape[2] * 2]) for i in range(len(features))]
            video_grid_thw = [[c_t, thw[1], thw[2]] for thw, combine_t in zip(video_grid_thw, combine_t_list) for c_t in combine_t]
            videos_inputs["video_grid_thw"] = torch.tensor(video_grid_thw)

        if not isinstance(text, list):
            text = [text]

        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while "<|image_pad|>" in text[i]:
                    text[i] = text[i].replace(
                        "<|image_pad|>", "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|image_pad|>")

        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            video_index = 0
            for i in range(len(text)):
                while "<|video_pad|>" in text[i]:
                    if timestamps is not None:
                        if combine_t_list is None:
                            t, h, w = video_grid_thw[video_index]
                            replacement = ""
                            for timestamp in timestamps[index]:
                                replacement += timestamp + "<|vision_start|>" + "<|placeholder|>" * (h * w // merge_length) + "<|vision_end|>"
                            text[i] = text[i].replace("<|video_pad|>", replacement, 1)
                            text[i] = text[i].replace("<|vision_start|>", "", 1)
                            text[i] = text[i].replace("<|vision_end|><|vision_end|>", "<|vision_end|>")
                            video_index += len(timestamps[index])
                        else:
                            t, h, w = video_grid_thw[video_index]
                            replacement = ""
                            for timestamp, combine_t in zip(timestamps[index], combine_t_list[index]):
                                replacement += timestamp + "<|vision_start|>" + "<|placeholder|>" * (combine_t * h * w // merge_length) + "<|vision_end|>"
                            text[i] = text[i].replace("<|video_pad|>", replacement, 1)
                            text[i] = text[i].replace("<|vision_start|>", "", 1)
                            text[i] = text[i].replace("<|vision_end|><|vision_end|>", "<|vision_end|>")
                            video_index += len(timestamps[index])
                    else:
                        text[i] = text[i].replace(
                            "<|video_pad|>", "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length), 1
                        )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|video_pad|>")

        _ = output_kwargs["text_kwargs"].pop("padding_side", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})