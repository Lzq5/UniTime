from __future__ import annotations

import base64
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO
import numpy as np
import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
# import cv2
import decord
import random
logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 128 * 2
VIDEO_TOTAL_PIXELS = 1024 * 16 * 28 * 28
MAX_FRAMES = 1024

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    # import ipdb;ipdb.set_trace()
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        nframes = min(max(nframes, min_frames), max_frames)
        nframes = round_by_factor(nframes, FRAME_FACTOR)
    return nframes

def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None

def _read_video_decord(
    ele: dict,
) -> torch.Tensor:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()

    video_start = ele.get("video_start", 0)
    video_end = ele.get("video_end", (total_frames - 1) / video_fps)
    start_frame = int(video_start * video_fps)
    end_frame = int(video_end * video_fps)
    start_frame_idx = max(0, start_frame)
    end_frame_idx = min(total_frames - 1, end_frame)

    total_frames_segment = end_frame_idx - start_frame_idx + 1
    fps = ele.get("fps", FPS)
    nframes = max(round_by_factor(int(total_frames_segment / video_fps * fps), FRAME_FACTOR), 2)
    return vr, nframes, start_frame_idx, end_frame_idx


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_decord_available():
        video_reader_backend = "decord"
    return video_reader_backend


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()
        vr, nframes_2fps, start_frame_idx, end_frame_idx = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        total_frames, video_fps = len(vr), vr.get_avg_fps()

        video_sample = vr.get_batch([0]).asnumpy()
        video_sample = torch.tensor(video_sample).permute(0, 3, 1, 2)
        _, _, height, width = video_sample.shape

        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes_2fps * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels = ele.get("max_pixels", max_pixels)
        if "feature" in ele:
            feat_path = ele["feature"]
            feature_data = torch.load(feat_path, map_location='cpu')
            feature = feature_data["feature"]
            fps_sample_feature_frame_idx = feature_data["frame_idx"]
            sample_fps = feature_data["sample_fps"]

            start_closest_idx = (fps_sample_feature_frame_idx - start_frame_idx).abs().argmin()
            end_closest_idx = (fps_sample_feature_frame_idx - end_frame_idx).abs().argmin()

            if "dynamic" in feat_path:
                feature_sample_idx = list(range(start_closest_idx, end_closest_idx+1))
            else:
                total_fps_sample_frames = end_closest_idx - start_closest_idx + 1
                if total_fps_sample_frames > int(MAX_FRAMES * sample_fps / 2):
                    feature_sample_idx = torch.linspace(start_closest_idx, end_closest_idx, int(MAX_FRAMES * sample_fps / 2)).round().long().tolist()
                else:
                    feature_sample_idx = list(range(start_closest_idx, end_closest_idx+1))
            
            idx = [fps_sample_feature_frame_idx[i] for i in feature_sample_idx]
            sampled_timestamps = [round(i.item() / video_fps, 1) for i in idx]
            feature = feature[feature_sample_idx]
            if feature.dim() != 4:
                feature = feature.unsqueeze(1).unsqueeze(1)

            num_clips = ele["num_clips"]
            clip_length = int(ele["clip_length"] * sample_fps / 2)

            feature, sampled_timestamps_combine, combine_t_list = combine_timestamps(feature, sampled_timestamps, num_clips, clip_length)
            return None, feature, sampled_timestamps_combine, combine_t_list
        else:
            if "resized_height" in ele and "resized_width" in ele:
                resized_height, resized_width = smart_resize(
                    ele["resized_height"],
                    ele["resized_width"],
                    factor=image_factor,
                )
            else:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=image_factor,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )

            nframes = total_pixels // (resized_height * resized_width) * FRAME_FACTOR
            nframes = floor_by_factor(nframes, FRAME_FACTOR)
            idx = torch.linspace(start_frame_idx, end_frame_idx, nframes).round().long().tolist()
            video = vr.get_batch(idx).asnumpy()
            video = torch.tensor(video).permute(0, 3, 1, 2)
            sampled_timestamps = [round(i/video_fps, 1) for i in idx]

            if nframes == 0:
                print(ele)

            video = transforms.functional.resize(
                video,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()
            combine_t_list = generate_clip_lengths(video.shape[0] // 2, 1)
            return video, None, sampled_timestamps, combine_t_list
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        return images
    
def generate_clip_lengths(t, clip_length):
    full_clips = t // clip_length
    remainder = t % clip_length
    result = [clip_length] * full_clips
    if remainder > 0:
        result.append(remainder)
    
    return result

def combine_timestamps(feature, sampled_timestamps, num_clips=32, clip_length=-1):
    T,H,W,D = feature.shape
    assert len(sampled_timestamps) == T
    if clip_length == -1:
        clip_length = T // num_clips
        
    sampled_timestamps_combine = sampled_timestamps[::int(clip_length)]
    combine_t_list = generate_clip_lengths(feature.shape[0], clip_length)

    return feature, sampled_timestamps_combine, combine_t_list

def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None]:
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    sampled_timestamps_list = []
    feature_inputs = []
    combine_t_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video, feature, sampled_timestamps, combine_t = fetch_video(vision_info)
            if video is not None:
                video_inputs.append(video)
            if feature is not None:
                feature_inputs.append(feature)
            if sampled_timestamps is not None:
                sampled_timestamps_list.append(sampled_timestamps)
            if combine_t is not None:
                combine_t_list.append(combine_t)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if len(feature_inputs) == 0:
        feature_inputs = None
    if len(sampled_timestamps_list) == 0:
        sampled_timestamps_list = None
    if len(combine_t_list) == 0:
        combine_t_list = None
    return image_inputs, video_inputs, sampled_timestamps_list, feature_inputs, combine_t_list
