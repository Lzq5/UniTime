import os
import json

MAXFRAME = 128

def save_json(content, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        f.write(json.dumps(content))

def load_data_to_dict(data_list, video_root, feat_root):
    qid = 0
    vc_dict = {}
    for data in data_list:
        for clips in data['clips']:
            clip_uid = clips['clip_uid']
            if os.path.exists(os.path.join(video_root, f"{clip_uid}.mp4")):
                if clip_uid not in vc_dict:
                    vc_dict[clip_uid] = {
                        "qid": qid,
                        "id": f"{clip_uid}", 
                        "annos": [], 
                        "duration": clips['clip_end_sec'] - clips['clip_start_sec'],
                        "video_path": os.path.join(video_root, f"{clip_uid}.mp4"),
                        "feature_path": os.path.join(feat_root, f"{clip_uid}.pt"),
                    }
                    qid += 1
                for annotations in clips['annotations']:
                    for anno in annotations['language_queries']:
                        if 'query' not in anno.keys():
                            continue
                        vc_dict[clip_uid]["annos"].append({
                            "query": anno['query'], 
                            "window": [[anno['clip_start_sec'], anno['clip_end_sec']]], 
                        })
            else:
                print(f"Video not found: {clip_uid}")
        
    return vc_dict

def build_test_data(test_dict):
    save_data = []
    qid = 0    
    for clip_uid, clip_data in test_dict.items():
        for anno in clip_data['annos']:
            if clip_data['duration'] >= MAXFRAME:
                mode = 'mr_seg'
            else:
                mode = 'mr'
            save_data.append({
                "qid": qid,
                "id": clip_data['id'],
                "annos": [anno],
                "duration": clip_data['duration'],
                "mode": mode,
            })
            qid += 1
    return save_data

def build_train_data(train_dict, repeat_num=4, clip_lengths=[96, 64, 32], annos_num=20):
    save_data = []
    qid = 0
    
    for clip_uid, clip_data in train_dict.items():
        if clip_data['duration'] >= MAXFRAME:
            mode = 'mr_seg'
        else:
            mode = 'mr'
        for _ in range(repeat_num):
            save_data.append({
                "qid": qid,
                "id": clip_data['id'],
                "annos": clip_data['annos'].copy(),
                "duration": clip_data['duration'],
                "mode": mode,
            })
            qid += 1
    
    for clip_uid, clip_data in train_dict.items():
        if clip_data['duration'] >= MAXFRAME:
            duration = clip_data['duration']
            annos = clip_data['annos']
            
            for clip_l in clip_lengths:
                clip_start = [i * clip_l for i in range(int(duration // clip_l))]
                clip_end = [min(i + clip_l, duration) for i in clip_start]
                
                for video_start, video_end in zip(clip_start, clip_end):
                    annos_clip = []
                    
                    for anno in annos:
                        if anno["window"][0][0] < video_end and anno["window"][0][1] > video_start:
                            annos_clip.append({
                                "query": anno['query'], 
                                "window": [[
                                    max(anno["window"][0][0], video_start), 
                                    min(video_end, anno["window"][0][1])
                                ]], 
                            })
                    
                    if annos_clip:
                        save_data.append({
                            "qid": qid,
                            "id": clip_data['id'],
                            "video_start": video_start,
                            "video_end": video_end,
                            "annos": annos_clip,
                            "duration": video_end - video_start,
                            "mode": 'mr',
                        })
                        qid += 1
    
    processed_data = []
    for anno_data in save_data:
        annos = anno_data['annos']
        if len(annos) == 0:
            continue
        elif len(annos) > annos_num:
            for i in range(0, len(annos), annos_num):
                new_annos = annos[i:i+annos_num]
                new_data = anno_data.copy()
                new_data['annos'] = new_annos
                new_data['qid'] = anno_data['qid'] if i == 0 else qid
                processed_data.append(new_data)
                if i != 0:
                    qid += 1
        else:
            processed_data.append(anno_data)
    
    return processed_data

def main():
    video_root = 'path_to_video_root' #[ToModify]
    feat_root = 'path_to_feature_folder' #[ToModify]
    ann_root = './'
    repeat_num = 4 #[ToModify] 4 for long video datasets, 1 for short video datasets
    annos_num = 45 #[ToModify] The number of QA samples for video-centric training can be adjusted based on your actual GPU memory.
    
    print("Processing training data...")
    raw_data = json.load(open('./nlq_train.json'))['videos']
    train_dict = load_data_to_dict(raw_data, video_root, feat_root)
    train_data = build_train_data(train_dict, repeat_num=repeat_num, annos_num=annos_num)
    save_json(train_data, os.path.join(ann_root, f'ego4d_train_uni_{repeat_num}.json'))
    print(f"Training data saved: {len(train_data)} samples")
    
    print("Processing test data...")
    raw_data = json.load(open('./nlq_val.json'))['videos']
    test_dict = load_data_to_dict(raw_data, video_root, feat_root)
    test_data = build_test_data(test_dict)
    save_json(test_data, os.path.join(ann_root, 'ego4d_test_all.json'))
    print(f"Test data saved: {len(test_data)} samples")

if __name__ == "__main__":
    main()
