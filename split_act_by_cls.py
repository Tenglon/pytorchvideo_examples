import numpy as np
import json
import os
from pathlib import Path
import shutil
from tqdm import tqdm

annotation_path = Path("./activity_net.v1-3.min.json")
src_base = Path('/local/tlong/act_net_trimmed/train')
out_base = Path('/local/tlong/act_train/train/')

def get_video_ids_annotations_and_fps(data):
    video_ids = []
    annotations = []

    for key, value in data['database'].items():

        video_ids.append(key)
        annotations.append(value['annotations'])

    return video_ids, annotations

def get_class_labels(data):
    class_names = []
    for node1 in data['taxonomy']:
        is_leaf = True
        for node2 in data['taxonomy']:
            if node2['parentId'] == node1['nodeId']:
                is_leaf = False
                break
        if is_leaf:
            class_names.append(node1['nodeName'])

    class_labels_map = {}

    for i, class_name in enumerate(class_names):
        class_labels_map[class_name] = i

    return class_labels_map


with annotation_path.open('r') as f:
    data = json.load(f)

video_ids, annotations = get_video_ids_annotations_and_fps(data)
print (len(video_ids))
print (len(list(set(video_ids))))
class_to_idx = get_class_labels(data)
idx_to_class = {}


for name, label in class_to_idx.items():
    idx_to_class[label] = name
    directory = out_base / name
    directory.mkdir(parents = True, exist_ok=True)





for i in tqdm(range(len(annotations))):
    if len(annotations[i])==0:
        continue
    else:
        for j, anno_item in enumerate(annotations[i]):
            src_path = src_base / f'{video_ids[i]}_{j}.mp4'
            dst_path = out_base / anno_item['label'] / f'{video_ids[i]}_{j}.mp4'

            if src_path.exists():

                if not dst_path.exists():

                    shutil.move(src_path, dst_path)