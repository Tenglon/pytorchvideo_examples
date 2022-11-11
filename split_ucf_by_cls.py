
from pathlib import Path
import os
from tqdm import tqdm

base = Path('/local/tlong/ucf101')
def link_video(dest, file_path:Path):
    video_fn = file_path.parts[-1]
    cls_name = video_fn.split('_')[1]
    out_base = dest / cls_name
    out_base.mkdir(parents=True, exist_ok=True)

    cmd = f'ln -s {video_fn} {out_base / video_fn}'
    os.system(cmd)

# for i in range(1,4):
if True:

    i = 1

    testfile_csv = Path(f'/local/tlong/ucf_anno/test0{i}.csv')
    trainfile_csv = Path(f'/local/tlong/ucf_anno/trainlist0{i}.csv')
    
    with open(testfile_csv, 'r') as fp:
        lines = fp.readlines()
        test_files = [line.split(' ')[0] for line in lines]
        test_files = [Path(item) for item in test_files]
    with open(trainfile_csv, 'r') as fp:
        lines = fp.readlines()
        train_files = [line.split(' ')[0] for line in lines]
        train_files = [Path(item) for item in train_files]

    for test_file in tqdm(test_files):
        # On purpose made dest to be train
        dest = Path(f'/local/tlong/ucf101_split{i}/train')
        dest.mkdir(parents=True,exist_ok=True)
        link_video(dest, test_file)

    for train_file in tqdm(train_files):
        dest = Path(f'/local/tlong/ucf101_split{i}/train')
        dest.mkdir(parents=True,exist_ok=True)
        link_video(dest, train_file)

    # import pdb
    # pdb.set_trace()
    
