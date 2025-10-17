import torch
import torch.nn.functional as F
import os
import decord
from collators.qwen_vision_process import smart_resize, IMAGE_FACTOR, round_by_factor, FRAME_FACTOR, VIDEO_MAX_PIXELS, VIDEO_MIN_PIXELS, FPS, floor_by_factor, generate_clip_lengths
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def resize_feature(feature, resize_h, resize_w):
    feature = feature.permute(0, 3, 1, 2)
    feature_resized = F.interpolate(feature, size=(resize_h, resize_w), mode='bilinear', align_corners=False)
    feature_resized = feature_resized.permute(0, 2, 3, 1)
    return feature_resized

def feature(model, processor, video_path, feature_root):
    os.makedirs(feature_root, exist_ok=True)
    with torch.no_grad():
        vid = os.path.splitext(os.path.basename(video_path))[0]
        visual_feature_path = f"{feature_root}/{vid}.pt"
        if os.path.exists(visual_feature_path):
            # print(visual_feature_path)
            try:
                feature_data = torch.load(visual_feature_path, map_location='cpu')
                return visual_feature_path
            except Exception as e:
                print(f"load error: {e}, re-extracting...")
        
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        except:
            print(f"errors in loading video {video_path}")
            return None
        total_frames, video_fps = len(vr), vr.get_avg_fps()

        video_sample = vr.get_batch([0]).asnumpy()
        video_sample = torch.tensor(video_sample).permute(0, 3, 1, 2)
        _, _, height, width = video_sample.shape

        # frame_idx = [i for i in range(0, len(vr), int(video_fps / sample_fps))]
        nframes_2fps = round_by_factor(int(total_frames / video_fps * FPS), FRAME_FACTOR)

        video_total_pixels = 1024 * 16 * 28 * 28
        video_min_pixels = 16 * 28 * 28
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
            return None

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
        return visual_feature_path
