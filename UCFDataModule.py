import pytorch_lightning
import pytorchvideo.data
import pytorchvideo.transforms
import torchvision
import torch.utils.data


class UCFDataModule(pytorch_lightning.LightningDataModule):

    def __init__(self,
                    data_path = '/home/longteng/ssd/ucf_anno/', # Dataset configuration
                    clip_duration = 16/30, # Duration of sampled clip for each video
                    batch_size = 8,
                    num_workers = 0, # Number of parallel processes fetching data
                    clip_frames = 16,
                    transform = None
                ):

        super().__init__()

        self._DATA_PATH = data_path 
        self._CLIP_DURATION = clip_duration
        self._BATCH_SIZE = batch_size
        self._NUM_WORKERS = num_workers  
        self._CLIP_FRAMES = clip_frames

        import pdb
        pdb.set_trace()

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
        train_dataset = pytorchvideo.data.Ucf101(
            data_path=self._DATA_PATH + 'trainlist01.csv',
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "random", self._CLIP_DURATION),
            decode_audio=False,
            transform=self._TRANSFORM,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            drop_last=True
        )

    def valid_dataloader(self):
        """
        Create the Kinetics validation partition from the list of video labels
        in {self._DATA_PATH}/val
        """
        val_dataset = pytorchvideo.data.Ucf101(
            data_path=self._DATA_PATH + 'test01.csv',
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "uniform", self._CLIP_DURATION),
            decode_audio=False,
            transform=self._TRANSFORM,
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            # shuffle=True,
            drop_last=True
        )


if __name__ == '__main__':
    data_module = UCFDataModule()
    train_loader = data_module.train_dataloader()
    valid_loader = data_module.valid_dataloader()

    for batch in train_loader:
        break
