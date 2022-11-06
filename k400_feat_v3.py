import torch
import numpy as np
from K400DataModule import K400DataModule
from torchvision.models.feature_extraction import create_feature_extractor

from tqdm import tqdm
from model import load_model
from pathlib import Path

# CUDA_VISIBLE_DEVICES=0 && python yk400_feat.py

class ARGS():
    def __init__(self):
        # data parameters
        self.dataset                    = 'kinetics'
        self.data_dir                   = '/local/tlong/yk400'
        self.mode                       = 'train' # mode here means train/val/test split

        self.num_sec_aud                = 1 # default
        self.aud_sample_rate            = 24000 # default
        self.aud_spec_type              = 2
        self.use_volume_jittering       = False # default
        self.use_audio_temp_jittering   = False # default
        self.z_normalize                = True  # default for audio

        self.batch_size                 = 10
        self.workers                    = 0

        # model parameters
        self.vid_base_arch = 'r2plus1d_18'
        self.aud_base_arch = 'resnet9'
        self.norm_feat     = True
        self.use_mlp       = True
        self.headcount     = 10
        self.num_clusters  = 400

        # feature parameters
        self.weights_path  = Path('/local/tlong/code/selavi/weights/selavi_kinetics.pth') # should be str type
        self.a_layer_names = ['base.layer1', 'base.layer2', 'base.layer3', 'base.layer4', 'base.flatten']
        self.v_layer_names = ['base.layer4', 'base.flatten']
        self.feat_base     = Path('./k400_feat/train')

def load_model_parameters(model, model_weights):
    loaded_state = model_weights
    self_state = model.state_dict()
    for name, param in loaded_state.items():
        param = param
        if 'module.' in name:
            name = name.replace('module.', '')
        if name in self_state.keys():
            self_state[name].copy_(param)
        else:
            print("didnt load ", name)


def define_and_load_model(args):
    model = load_model(
        vid_base_arch = args.vid_base_arch,
        aud_base_arch = args.aud_base_arch,
        norm_feat     = args.norm_feat,
        use_mlp       = args.use_mlp,
        headcount     = args.headcount,
        num_classes   = args.num_clusters,
    )

    print("Loading model weights")
    if args.weights_path.exists():
        ckpt_dict = torch.load(args.weights_path)
        model_weights = ckpt_dict["model"]
        epoch = ckpt_dict["epoch"]
        print(f"Epoch checkpoint: {epoch}")
        load_model_parameters(model, model_weights)
    else:
        print("Random weights")
    return model

def get_clip_feat_path(feat_base, video_name, clip_index, i):

    path_parts = Path(video_name[i]).parts
    class_name, video_fn = path_parts[-2], path_parts[-1], 
    video_base = feat_base / class_name / video_fn

    video_base.mkdir(parents=True, exist_ok=True)

    # output file name is defined as {video_name}/{clip_index}.npz
    clip_feat_fn_wo_ext = video_base / str(clip_index[i].item())
    clip_feat_fn_wi_ext = clip_feat_fn_wo_ext.with_suffix('.npz')
    return clip_feat_fn_wi_ext

def get_clip_feat(args, i, v_feats, a_feats, batch_label):

    out_feat_dict = dict()
    out_feat_dict['label'] = batch_label

    for layer_name in args.a_layer_names:
        a_key = 'audio.' + layer_name
        out_feat_dict[a_key] = a_feats[layer_name][i].unsqueeze(dim = 0).numpy()

    for layer_name in args.v_layer_names:
        v_key = 'video.' + layer_name
        out_feat_dict[v_key] = v_feats[layer_name][i].unsqueeze(dim = 0).numpy()

    return out_feat_dict

if __name__ == '__main__':
    # Step0 : Degine args and data module
    ################################################################################################
    args = ARGS()
    data_module = K400DataModule(args.data_dir,
                                 batch_size = args.batch_size,
                                 num_workers = args.workers)
    # train_loader = data_module.train_dataloader()
    # valid_loader = data_module.val_dataloader()
    feat_train_loader = data_module.feat_train_dataloader()
    ################################################################################################

    # Step1: define model and load pre-trianed weights
    ################################################################################################
    model = define_and_load_model(args)
    torch.cuda.empty_cache()
    model.eval()
    model = model.cuda()
    ################################################################################################

    # Step2: define feature_extractors and prepare output parameters
    ################################################################################################
    feature_extractor_audio = create_feature_extractor(model.audio_network, return_nodes = args.a_layer_names)
    feature_extractor_video = create_feature_extractor(model.video_network, return_nodes = args.v_layer_names)

    feat_base = args.feat_base
    feat_base.mkdir(parents=True, exist_ok=True)

    ################################################################################################

    # Step3: extract features
    ################################################################################################
    for batch in tqdm(feat_train_loader):

        if 'audio' not in batch.keys():
            continue

        frames, audio, batch_label = batch['video'], batch['audio'], batch['label']
        video_name, video_index, clip_index = batch['video_name'], batch['video_index'], batch['clip_index']

        # Step3.1: check if entire batch are already extracted, to avoid forward() 
        ################################################################################################
        exist_samples_in_batch = set()
        for i in range(len(batch_label)):
            
            clip_feat_path = get_clip_feat_path(feat_base, video_name, clip_index, i)

            if clip_feat_path.exists():
                # print(f'video {video_name[i]}, clip {clip_index[i]} feature exists')
                exist_samples_in_batch.add(i)
                continue

        if len(exist_samples_in_batch) == range(len(batch_label)):
            # print(f'skip batch for video {video_name[i]} as feature exists')
            continue
        ################################################################################################

        # Step3.2 Forward pass to get the features
        ################################################################################################
        frames, audio = frames.cuda(), audio.cuda()

        v_feats_gpu = feature_extractor_video(frames)
        a_feats_gpu = feature_extractor_audio(audio)

        v_feats = {key: value.detach().cpu() for key, value in v_feats_gpu.items()}
        a_feats = {key: value.detach().cpu() for key, value in a_feats_gpu.items()}
        del v_feats_gpu, a_feats_gpu
        ################################################################################################

        # Step3.3 Loop over each sample in batch to store the features
        ################################################################################################
        no_exist_samples_in_batch = set(range(len(batch_label))) - exist_samples_in_batch
        save_processes = []
        for i in no_exist_samples_in_batch:

            print(f'processing video {video_name[i]}, clip {clip_index[i]}')

            clip_feat_path = get_clip_feat_path(feat_base, video_name, clip_index, i)

            # feature exists, next sample
            if clip_feat_path.exists():
                continue

            # feature not exist, store itb
            out_feat_dict = get_clip_feat(args, i, v_feats, a_feats, batch_label)
            # torch.save(out_feat_dict, clip_feat_path)   # type: ignore
            np.savez(clip_feat_path, out_feat_dict)   # type: ignore
            # np.savez_compressed(clip_feat_path, out_feat_dict)   # type: ignore
        ################################################################################################

