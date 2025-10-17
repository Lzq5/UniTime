import torch
from models.qwen2_vl import Qwen2VLMRForConditionalGeneration, Qwen2VLMRProcessor
from collators.qwen_vision_process import generate_clip_lengths
import torch.nn.functional as F
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./datasets/ego4d/val.json', type=str)
parser.add_argument('--model_local_path', type=str, default='Path to the Qwen2-VL', help='Path to the local model')
parser.add_argument('--video_root', type=str, help='Path to the video root')
parser.add_argument('--feat_root', type=str, default='./tmp_feature')
parser.add_argument('--part', default=0, type=int)
parser.add_argument('--num_parts', default=8, type=int)
parser.add_argument('--gpu', default=0, type=int)
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}')
data_path = args.data_path

feature_root = args.feat_root
video_root = args.video_root

if 'ego4d' in data_path or 'nlq' in data_path:
    dataset = 'ego4d'
    video_type = 'mp4'
elif 'tacos' in data_path:
    dataset = 'tacos'
    video_type = 'avi'

feature_path = f'{feature_root}/{dataset}'

import os
os.makedirs(feature_path, exist_ok=True)

print(f"processing {dataset} part {args.part}")

compute_dtype = torch.bfloat16
device_map = None
bnb_config = None
model_local_path = args.model_local_path
loading_kwargs = dict(
        torch_dtype=compute_dtype,
        quantization_config=bnb_config,
        device_map=device_map,
    )

model = Qwen2VLMRForConditionalGeneration.from_pretrained(
        model_local_path,
        attn_implementation="flash_attention_2",
        **loading_kwargs,
    )
processor = Qwen2VLMRProcessor.from_pretrained(model_local_path)

def resize_feature(feature, resize_h, resize_w):
    feature = feature.permute(0, 3, 1, 2)
    feature_resized = F.interpolate(feature, size=(resize_h, resize_w), mode='bilinear', align_corners=False)
    feature_resized = feature_resized.permute(0, 2, 3, 1)
    return feature_resized

with torch.no_grad():
    model.to(device)
    model.eval()

    import json
    list_data_dict = json.load(open(data_path, "r"))

    vid_list = []
    video_path_list = []
    BATCH_SIZE = 8
    
    for i in range(len(list_data_dict)):
        source = list_data_dict[i]
        vid = source["id"]
        if source["mode"] == 'mr':
            continue
        visual_feature_path = f"{feature_path}/{vid}.pt"
        if os.path.exists(visual_feature_path):
            continue
        if vid not in vid_list:
            vid_list.append(vid)
            video_path_list.append(source.get("video_path", None))
    
    total_data = len(vid_list)
    num_parts = args.num_parts
    part_size = total_data // num_parts
    start_idx = args.part * part_size
    if args.part == num_parts - 1:
        end_idx = total_data
    else:
        end_idx = (args.part + 1) * part_size
    vid_list_subset = vid_list[start_idx:end_idx]
    video_path_list_subset = video_path_list[start_idx:end_idx]

    import decord
    from tqdm import tqdm
    from collators.qwen_vision_process import smart_resize, IMAGE_FACTOR, round_by_factor, FRAME_FACTOR, VIDEO_MAX_PIXELS, VIDEO_MIN_PIXELS, FPS, floor_by_factor
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    for i in tqdm(range(len(vid_list_subset))):
        vid = vid_list_subset[i]
        visual_feature_path = f"{feature_path}/{vid}.pt"
        if os.path.exists(visual_feature_path):
            try:
                old_feature = torch.load(visual_feature_path)["feature"]
                old_t, old_h, old_w, _ = old_feature.shape
                if old_t * old_h * old_w <= 1024 * 16:
                    continue
                else:
                    print("re extract feature")
            except:
                print("wrong feature file")
        
        video_path = video_path_list_subset[i]
        if video_path == None:
            video_path = os.path.join(video_root,f"{vid}.{video_type}")
        
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        except:
            print(vid, video_path)
            continue
        total_frames, video_fps = len(vr), vr.get_avg_fps()

        video_sample = vr.get_batch([0]).asnumpy()
        video_sample = torch.tensor(video_sample).permute(0, 3, 1, 2)
        _, _, height, width = video_sample.shape

        nframes_2fps = round_by_factor(int(total_frames / video_fps * FPS), FRAME_FACTOR)

        video_total_pixels = 1024 * 16 * 28 * 28
        video_min_pixels = 32 * 28 * 28
        video_max_pixels = 768 * 28 * 28
        image_factor = 28

        max_pixels = max(min(video_max_pixels, video_total_pixels / nframes_2fps * FRAME_FACTOR), int(video_min_pixels))

        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=IMAGE_FACTOR,
            min_pixels=VIDEO_MIN_PIXELS,
            max_pixels=VIDEO_MAX_PIXELS,
        )

        new_resized_height, new_resized_width = smart_resize(
            height,
            width,
            factor=IMAGE_FACTOR,
            min_pixels=video_min_pixels,
            max_pixels=max_pixels,
        )

        if max_pixels == video_min_pixels:
            nframes = video_total_pixels // max_pixels * FRAME_FACTOR
        else:
            nframes = video_total_pixels // (new_resized_height * new_resized_width) * FRAME_FACTOR

        nframes = floor_by_factor(nframes, FRAME_FACTOR)
        frame_idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
        sample_fps = nframes / total_frames * video_fps
        fps_sample_feature_frame_idx = [int((x + y)/2) for x, y in zip(frame_idx[::2], frame_idx[1::2])]
        total_sample_frames = len(frame_idx)
        

        try:
            frames = vr.get_batch(frame_idx).asnumpy()
        except:
            print(vid, video_path)
            continue

        video = torch.tensor(frames).permute(0, 3, 1, 2)
        video_inputs = [transforms.functional.resize(
                video,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()]
        
        inputs = processor(
            text=['hello'],
            images=None,
            videos=video_inputs,
            timestamps=None,
            padding=True,
            return_tensors="pt",
        )

        pixel_values_videos = inputs['pixel_values_videos']
        video_grid_thw = inputs['video_grid_thw']

        pixel_values_videos = pixel_values_videos.type(model.visual.get_dtype()).to(model.device)
        combine_t_list = [generate_clip_lengths(video.shape[0] // 2, 1)]
        video_embeds = model.encode_video_chunk(pixel_values_videos, video_grid_thw, combine_t_list).cpu()
        video_embeds = video_embeds.reshape(len(combine_t_list[0]), video_grid_thw[0][1] // 2, video_grid_thw[0][2] // 2, video_embeds.shape[-1])
        video_embeds_resize = resize_feature(video_embeds, resize_h=new_resized_height//28, resize_w=new_resized_width//28)
        torch.save({"feature":video_embeds_resize, "frame_idx": torch.tensor(fps_sample_feature_frame_idx), "sample_fps":sample_fps}, visual_feature_path)