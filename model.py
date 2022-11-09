# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from collections import OrderedDict

from small_feat_loader import SmallFeatDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Flatten(nn.Module):
    """A shape adaptation layer to patch certain networks."""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Unsqueeze(nn.Module):
    """A shape adaptation layer to patch certain networks."""
    def __init__(self):
        super(Unsqueeze, self).__init__()

    def forward(self, x):
        return x.unsqueeze(-1)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


def random_weight_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None: 
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class MLPv2(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.3):
        super(MLPv2, self).__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes, bias=True)
            )
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                Unsqueeze(),
                nn.BatchNorm1d(n_hidden),
                Flatten(),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True)
            )

    def forward(self, x):
        return self.block_forward(x)

# Get the layers after feat_name layer.
def get_part_model(feat_name, full_model):

    block_names = [block[0] for block in full_model.named_children()]
    blocks      = [block[1] for block in full_model.named_children()]

    v_feat_layer = feat_name.split('.')[-1]
    pos = block_names.index(v_feat_layer)
    layer_dict = OrderedDict(zip(block_names[pos + 1:], blocks[pos + 1:]))

    part_model = nn.Sequential(layer_dict)

    return part_model

class VideoBaseNetwork(nn.Module):
    
    def __init__(self, v_feat_name, vid_base_arch='r2plus1d_18', pretrained=False, norm_feat=False):
        super(VideoBaseNetwork, self).__init__()

        full_model = torchvision.models.video.__dict__[vid_base_arch](pretrained=pretrained)
        if not pretrained:
            print("Randomy initializing models")
            random_weight_init(full_model)

        self.part_model = get_part_model(v_feat_name, full_model)

        self.part_model.fc = Identity()

        self.norm_feat = norm_feat

    def forward(self, x):
        x = self.part_model(x).squeeze()
        if self.norm_feat:
            x = F.normalize(x, p=2, dim=1)
        return x


class AudioBaseNetwork(nn.Module):

    def __init__(self, a_feat_name, aud_base_arch='resnet9', pretrained=False, norm_feat=False):

        super(AudioBaseNetwork, self).__init__()

        assert aud_base_arch=='resnet9'
        assert pretrained==False

        full_model = torchvision.models.resnet._resnet(torchvision.models.resnet.BasicBlock, 
            [1,1,1,1], weights = None, progress=False)


        # change Input to 1 channel 
        full_model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        
        self.part_model = get_part_model(a_feat_name, full_model)

        self.part_model.fc = Identity()  # type: ignore

        self.norm_feat = norm_feat

    def forward(self, x):
        x = self.part_model(x).squeeze()
        if self.norm_feat:
            x = F.normalize(x, p=2, dim=1)
        return x


class AVModel(nn.Module):
    def __init__(
        self,
        vid_base_arch='r2plus1d_18', 
        aud_base_arch='resnet9',
        v_feat_name='video.base.layer4', 
        a_feat_name='audio.base.layer1', 
        pretrained=False, 
        norm_feat=True, 
        use_mlp=False,
        headcount=1, 
        num_classes=256, 
    ):
        super(AVModel, self).__init__()

        # Save proprties
        self.use_mlp = use_mlp
        self.hc = headcount
        self.norm_feat = norm_feat
        self.return_features = False

        self.video_network = VideoBaseNetwork(
            v_feat_name, 
            vid_base_arch, 
            pretrained=pretrained
        )
        self.audio_network = AudioBaseNetwork(
            a_feat_name,
            aud_base_arch,
            pretrained=pretrained
        )
        encoder_dim = 512
        encoder_dim_a = 512
        n_hidden = 512

        if self.hc == 1:
            if use_mlp:
                print("Using MLP to be combined with SyncBN")
                self.mlp_v = MLPv2(encoder_dim, num_classes, n_hidden=n_hidden)
                self.mlp_a = MLPv2(encoder_dim_a, num_classes)
            else:
                print("Using Linear Layer")
                self.mlp_v = nn.Linear(encoder_dim, num_classes)
                self.mlp_a = nn.Linear(encoder_dim_a, num_classes)
        else:
            if use_mlp:
                print("Using MLP to be combined with SyncBN")
                for a, i in enumerate(range(self.hc)):
                    setattr(self, "mlp_v%d"%a, MLPv2(encoder_dim, num_classes, n_hidden=n_hidden))
                    setattr(self, "mlp_a%d"%a, MLPv2(encoder_dim_a, num_classes))
            else:
                for a, i in enumerate(range(self.hc)):
                    setattr(self, "mlp_v%d"%a, nn.Linear(encoder_dim, num_classes))
                    setattr(self, "mlp_a%d"%a, nn.Linear(encoder_dim_a, num_classes))


    def forward(self, img, spec, whichhead=0):
        img_features = self.video_network(img).squeeze()
        aud_features = self.audio_network(spec).squeeze()

        if self.return_features:
            return img_features, aud_features
        if len(aud_features.shape) == 1:
            aud_features = aud_features.unsqueeze(0)
        if len(img_features.shape) == 1:
            img_features = img_features.unsqueeze(0)

        if self.hc == 1:
            nce_img_features = self.mlp_v(img_features)
            nce_aud_features = self.mlp_a(aud_features)
            if self.norm_feat:
                nce_img_features = F.normalize(nce_img_features, p=2, dim=1)
                nce_aud_features = F.normalize(nce_aud_features, p=2, dim=1)
            return nce_img_features, nce_aud_features
        elif self.hc > 1:
            # note: will return lists here.
            outs1 = []
            outs2 = []
            for head in range(self.hc):
                img_f = getattr(self, "mlp_v%d"%head)(img_features)
                aud_f = getattr(self, "mlp_a%d"%head)(aud_features)
                if self.norm_feat:
                    img_f = F.normalize(img_f, p=2, dim=1)
                    aud_f = F.normalize(aud_f, p=2, dim=1)
                outs1.append(img_f)
                outs2.append(aud_f)
            return outs1, outs2


def load_model(
    vid_base_arch='r2plus1d_18', 
    aud_base_arch='resnet9',
    v_feat_name='video.base.layer4', 
    a_feat_name='audio.base.layer1', 
    pretrained=False, 
    norm_feat=True, 
    use_mlp=False,
    headcount=1, 
    num_classes=256, 
):  
    model = AVModel(
        vid_base_arch=vid_base_arch, 
        aud_base_arch=aud_base_arch,
        v_feat_name=v_feat_name, 
        a_feat_name=a_feat_name, 
        pretrained=pretrained, 
        norm_feat=norm_feat, 
        use_mlp=use_mlp,
        headcount=headcount, 
        num_classes=num_classes, 
    )
    return model

if __name__ == '__main__':

    a_feat_name = 'audio.base.layer1'
    v_feat_name = 'video.base.layer4'

    feat_base = '/home/longteng/ssd/k400_small_feat/train'
    training_data = SmallFeatDataset(feat_base, a_feat_name = a_feat_name, v_feat_name = v_feat_name)
    train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True, num_workers=20)

    model = load_model(
        v_feat_name=v_feat_name,
        a_feat_name=a_feat_name,
        use_mlp=False,
        headcount=1, 
        num_classes=256, 
    )

    for a_feat, v_feat, label in tqdm(train_dataloader):

        a_out = model.audio_network(a_feat)
        v_out = model.video_network(v_feat)

    pass
