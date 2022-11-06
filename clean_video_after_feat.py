# This script is used to help debugging 
from pathlib import Path
from glob import glob
import shutil


src_base = Path('/local/tlong/yk400')

dst_base = Path('/local/tlong/done_k400')
dst_base.mkdir(parents=True, exist_ok=True)

feat_base = Path('/local/tlong/code/selavi/k400_feat/train/')

video_feat_bases = feat_base.glob('y*/*.mp4')

count = 0

for video_feat_base in video_feat_bases:

    pth_list = list(video_feat_base.glob('*.npz'))

    if len(pth_list) == 10:

        parts = video_feat_base.parts
        split, class_name, video_fn = parts[-3], parts[-2], parts[-1]

        src_clsbase = src_base / split / class_name
        dst_clsbase = dst_base / split / class_name

        # src_clsbase.mkdir(parents=True, exist_ok=True)
        dst_clsbase.mkdir(parents=True, exist_ok=True)
        
        src_path = src_clsbase / video_fn
        dst_path = dst_clsbase / video_fn 

        if src_path.exists():
            shutil.move(src_path, dst_path)
            count += 1

print(f'a total of {count} videos are removed from waiting list')
