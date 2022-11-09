import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class SmallFeatDataset(Dataset):

    def __init__(self, feat_base, a_feat_name = 'audio.base.layer1', v_feat_name = 'video.base.layer4'):
        
        self.feat_base = Path(feat_base)
        self.feat_cls_bases = self.feat_base.glob('*') 
        logging.info('Load audio feat paths')
        self.a_feat_paths = list(self.feat_base.glob(f'*/*/{a_feat_name}.npz'))
        logging.info('Load visual feat paths')
        self.v_feat_paths = list(self.feat_base.glob(f'*/*/{v_feat_name}.npz'))
        logging.info('Load paths Done')

        self.a_feat_paths.sort()
        self.v_feat_paths.sort()
        logging.info('Sort paths Done')

        self.label_set = [item.parts[-1] for item in self.feat_cls_bases]
        self.label_set.sort()

    def __len__(self):
        return len(self.a_feat_paths)

    def __getitem__(self, idx):

        a_npz_path = self.a_feat_paths[idx]
        v_npz_path = self.v_feat_paths[idx]

        with np.load(a_npz_path, allow_pickle = True) as data:
            a_feat = data['arr_0']

        with np.load(v_npz_path, allow_pickle = True) as data:
            v_feat = data['arr_0']

        assert a_npz_path.parts[-3] == v_npz_path.parts[-3]
        assert a_feat.shape[0] == v_feat.shape[0]

        cls_name = a_npz_path.parts[-3]
        label = self.label_set.index(cls_name)
        n_sec = a_feat.shape[0]

        rand_inx = torch.randperm(n_sec)[0]

        return a_feat[rand_inx, :, :], v_feat[rand_inx, :, :], label

feat_base = '/home/longteng/ssd/k400_small_feat/train'
training_data = SmallFeatDataset(feat_base, a_feat_name = 'audio.base.layer1', v_feat_name = 'video.base.layer4')

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=20)

for a_feat, v_feat, label in tqdm(train_dataloader):

    pass
