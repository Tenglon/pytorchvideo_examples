from UCFDataModule import UCFDataModule

import pandas as pd
import torchvision
import pytorch_lightning
import pytorchvideo.models.r2plus1d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"])
        self.log("train_loss", loss.item(), batch_size=len(y_hat))

        return_dict = dict()
        return_dict['video_name']  = batch['video_name']
        return_dict['video_index'] = batch['video_index']
        return_dict['clip_index']  = batch['clip_index']
        return_dict['aug_index']   = batch['aug_index']
        return_dict['label']       = batch['label']
        return_dict['pred']        = y_hat
        return_dict['loss']        = loss

        # print('training_step')

        return return_dict

    def validation_step(self, batch, batch_idx):

        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"])
        self.log("val_loss", loss, batch_size=len(y_hat))

        return_dict = dict()

        return_dict['video_name']  = batch['video_name']
        return_dict['video_index'] = batch['video_index']
        return_dict['clip_index']  = batch['clip_index']
        return_dict['aug_index']   = batch['aug_index']
        return_dict['label']       = batch['label']
        return_dict['pred']        = y_hat
        return_dict['loss']        = loss

        # print('validation_step')

        return return_dict

    def training_step_end(self, batch_parts):
        # losses from each GPU
        losses = batch_parts["loss"]

        new_batch_parts = batch_parts
        new_batch_parts['loss'] = losses.mean()

        return new_batch_parts

    def validation_step_end(self, batch_parts):
        # losses from each GPU
        losses = batch_parts["loss"]

        new_batch_parts = batch_parts
        new_batch_parts['loss'] = losses.mean()

        # print('validation_step_end')

        return new_batch_parts

    def epoch_end_eval(self, outputs, flag = 'training'):
        video_name_list, video_index_list = [], []
        clip_index_list, aug_index_list, label_list = [], [], []
        pred_list, loss_list = [], []

        batch_size = len(outputs[0]['label'])

        for batch in outputs:
            
            # video_name_list.extend(batch['video_name'])
            video_index_list.extend(batch['video_index'].cpu())
            clip_index_list.extend(batch['clip_index'].cpu())
            # aug_index_list.extend(batch['aug_index'].cpu())
            label_list.extend(batch['label'].cpu())

            pred_list.append(batch['pred'].cpu())
            loss_list.append(batch['loss'].cpu().repeat(batch_size))

        video_index_list2 = torch.hstack(video_index_list) # arrange pred for each sample in batch.
        clip_index_list2  = torch.hstack(clip_index_list)
        label_list2       = torch.hstack(label_list)

        pred_list2        = list(torch.vstack(pred_list).numpy())
        loss_list2        = torch.hstack(loss_list)

        epoch_df = pd.DataFrame.from_dict({
                                # 'video_name': video_name_list,
                                'video_index':video_index_list2,
                                'clip_index':clip_index_list2,
                                # 'aug_index':aug_index_list,
                                'label':label_list2,
                                'pred':pred_list2,
                                'loss':loss_list2
                                })

        # epoch_df.groupby('video_index')['pred'].agg(lambda x: np.stack(x).mean(axis=0))
        # epoch_df.groupby('video_index')['loss'].agg(lambda x: np.mean(x))

        output_df = epoch_df.groupby('video_index')[['clip_index',
                                       'label',
                                       'pred',
                                       'loss']].agg({'clip_index':lambda x:list(x),
                                                     'label': lambda x: list(x)[0],
                                                     'pred': lambda x: np.stack(x).mean(axis=0),
                                                     'loss': lambda x: np.mean(x) 
                                                    })

        output_df['y_hat'] = output_df['pred'].apply(np.argmax)

        epoch_loss = output_df['loss'].mean(axis=0)
        epoch_acc = (output_df['y_hat'] == output_df['label']).mean(axis=0)

        self.log(f'{flag}_loss', epoch_loss, batch_size=batch_size)
        self.log(f'{flag}_acc', epoch_acc, batch_size=batch_size)

    def training_epoch_end(self, training_step_outputs):

        self.epoch_end_eval(training_step_outputs, flag = 'training')  # type: ignore

    def validation_epoch_end(self, validation_step_outputs):

        self.epoch_end_eval(validation_step_outputs, flag = 'validation')  # type: ignore

    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr=1e-4)


if __name__ == '__main__':
    # model_r2plus1d = pytorchvideo.models.r2plus1d.create_r2plus1d(input_channel=3,
    #                                                         model_depth=18,
    #                                                         model_num_class=400,
    #                                                         norm=nn.BatchNorm3d,
    #                                                         activation=nn.ReLU,
    #                                                         )
    model_r2plus1d = torchvision.models.video.r2plus1d_18(num_classes = 102)

    model_r2plus1d = model_r2plus1d.cuda()
    print(model_r2plus1d)

    classification_module = VideoClassificationLightningModule(model_r2plus1d)

    data_module = UCFDataModule(num_workers = 0, batch_size= 12)

    trainer = pytorch_lightning.Trainer(accelerator="gpu", devices=1, strategy='dp')
    trainer.fit(classification_module, data_module)
