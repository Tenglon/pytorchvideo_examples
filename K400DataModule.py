import pytorch_lightning
import pytorchvideo.data
import pytorchvideo.transforms
import torchvision
import torch.utils.data
import torch.nn.functional as F
import os

from tqdm import tqdm
import numpy as np
import ffmpeg
from python_speech_features import logfbank


class K400DataModule(pytorch_lightning.LightningDataModule):

    def __init__(self,
                    data_path = '/home/longteng/ssd/Kinetics400split', # Dataset configuration
                    clip_duration = 1, # Duration of sampled clip for each video
                    batch_size = 4,
                    num_workers = 0, # Number of parallel processes fetching data
                    clip_frames = 30,
                    clip_sampler_type = 'random',
                    transform = None
                ):

        super().__init__()

        self._DATA_PATH = data_path 
        self._CLIP_DURATION = clip_duration
        self._BATCH_SIZE = batch_size
        self._NUM_WORKERS = num_workers  
        self._CLIP_FRAMES = clip_frames
        self._CLIP_SAMLPER = pytorchvideo.data.make_clip_sampler(clip_sampler_type, self._CLIP_DURATION)

        if transform is not None:
            self._TRANSFORM = transform
        else:
            self._TRANSFORM = torchvision.transforms.Compose([
                pytorchvideo.transforms.ApplyTransformToKey(
                    key="video",
                    transform=torchvision.transforms.Compose([
                        pytorchvideo.transforms.UniformTemporalSubsample(self._CLIP_FRAMES),
                        pytorchvideo.transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                        pytorchvideo.transforms.RandomShortSideScale(min_size=256, max_size=320),
                        torchvision.transforms.RandomCrop(224),
                        torchvision.transforms.RandomHorizontalFlip(p=0.5),
                        ])
                )
            ])

    def train_dataloader(self):
        """
        Create the Kinetics train partition from the list of video labels
        in {self._DATA_PATH}/train
        """
        train_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self._DATA_PATH, 'train'),
            clip_sampler=self._CLIP_SAMLPER,
            decode_audio=True,
            transform=self._TRANSFORM,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            # shuffle = True,
            drop_last=True
        )

    def val_dataloader(self):
        """
        Create the Kinetics validation partition from the list of video labels
        in {self._DATA_PATH}/val
        """
        val_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self._DATA_PATH, 'val'),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "uniform", self._CLIP_DURATION),
            decode_audio=True,
            transform=self._TRANSFORM,
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            shuffle=False,
            drop_last=False
        )

    def feat_train_dataloader(self):
        """
        Create the Kinetics validation partition from the list of video labels
        in {self._DATA_PATH}/val
        """
        feat_transform = torchvision.transforms.Compose([
                pytorchvideo.transforms.ApplyTransformToKey(
                    key="video",
                    transform=torchvision.transforms.Compose([
                        pytorchvideo.transforms.UniformTemporalSubsample(self._CLIP_FRAMES),
                        pytorchvideo.transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                        torchvision.transforms.Resize((224, 224)),
                        ])
                )
            ])
        
        feat_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self._DATA_PATH, 'train'),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "uniform", self._CLIP_DURATION),
            decode_audio=True,
            transform=feat_transform,
        )

        return torch.utils.data.DataLoader(
            feat_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            shuffle=False,
            drop_last=False
        )


if __name__ == '__main__':
    data_module = K400DataModule()
    # train_loader = data_module.train_dataloader()
    # valid_loader = data_module.val_dataloader()
    feat_train_loader = data_module.feat_train_dataloader()

    for batch in tqdm(feat_train_loader):
        frames, audio_raw, label = batch['video'], batch['audio'], batch['label']
        video_name, video_index, clip_index = batch['video_name'], batch['video_index'], batch['clip_index']

        import pdb  
        pdb.set_trace()

        break
        pass
