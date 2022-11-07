from pathlib import Path
import numpy as np
from tqdm import tqdm

feat_base = Path('./k400_feat/train/')

small_base = Path('./small_feat/train/')
small_base.mkdir(parents = True, exist_ok=True)

video_feat_paths = feat_base.glob('w*/*.mp4')

for video_feat_path in tqdm(video_feat_paths):

    npz_paths = list(video_feat_path.glob('*.npz'))
    npz_paths.sort()

    keys = ['audio.base.layer1', 'audio.base.layer2', 'audio.base.layer3', 'audio.base.layer4',
            'audio.base.flatten', 'video.base.layer4', 'video.base.flatten']

    output_dict = {key: [] for key in keys} # a dictionary of list, key is feat name and value is a list.

    for npz_path in npz_paths:

        with np.load(npz_path, allow_pickle = True) as data:

            feat_dict = data['arr_0'].item()
            for key in keys:
                output_dict[key].append(feat_dict[key])

    _, _, class_name, video_fn = video_feat_path.parts
    out_path = small_base / class_name / video_fn
    out_path.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        continue

    for key in keys:
        out_feat = np.vstack(output_dict[key])
        out_feat_path = (out_path / key)
        np.savez_compressed(out_feat_path, out_feat)
