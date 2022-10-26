
import os
import re
from glob import glob
from tqdm import tqdm
import pandas as pd

anno_base = '/local/tlong/data/ucf_anno'
class_ind_file = os.path.join(anno_base, 'classInd.txt')

with open(class_ind_file, 'r') as fp:
    lines = fp.readlines()
    lines = [line.rstrip() for line in lines]

    
# a dictionary {'applyeyemakers' : 1, ...}
class2ind = {line.split(' ')[1].lower() : line.split(' ')[0] for line in lines}

video_base = '/local/tlong/data/ucf101/'

video_pattern = os.path.join(video_base, '*.avi')

video_paths = glob(video_pattern)

path2clsind = dict()

for video_path in video_paths:
    
    clsname = os.path.split(video_path)[-1].split('_')[1]
    clsind  = class2ind[clsname.lower()]
    
    path2clsind[video_path] = clsind
path2clsind_df = pd.DataFrame.from_dict({'path':path2clsind.keys(), 'label':path2clsind.values()}, orient = 'columns')

train1_file = os.path.join(anno_base, 'trainlist01.txt')
train2_file = os.path.join(anno_base, 'trainlist02.txt')
train3_file = os.path.join(anno_base, 'trainlist03.txt')

for train_file in [train1_file, train2_file, train3_file]:

    with open(train_file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.rstrip() for line in lines]
        
    train_csv = train_file.replace('.txt', '.csv')

    with open(train_csv, 'w+') as fp:

        new_lines = []
        for line in lines:
            new_line = re.sub('[a-zA-Z0-9]*/', video_base, line)
            new_lines.append(new_line + '\n')
            
        fp.writelines(new_lines)


test1_file = os.path.join(anno_base, 'testlist01.txt')
test2_file = os.path.join(anno_base, 'testlist01.txt')
test3_file = os.path.join(anno_base, 'testlist01.txt')

for i,test_file in enumerate([test1_file, test2_file, test3_file]):

    test_dict = dict()

    with open(test_file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.rstrip() for line in lines]
        
    for line in tqdm(lines):
        v_fn = line.split('/')[1]
        v_path = os.path.join(video_base, v_fn)
        
        row_pos = path2clsind_df.path.str.contains(v_fn)
        label = path2clsind_df.loc[row_pos,'label']
        label = label.values[0]

        test_dict[v_path] = label

    test_path2clsind_df = pd.DataFrame.from_dict(test_dict, orient='index')

    target_path = os.path.join(anno_base, f'test0{i+1}.csv')
    test_path2clsind_df.to_csv(target_path, header=None, sep=' ')

