import torch
from K400DataModule import K400DataModule
from torchvision.models.feature_extraction import create_feature_extractor

from tqdm import tqdm
from model import load_model
from pathlib import Path

class ARGS():
    def __init__(self):
        # data parameters
        self.dataset                    = 'kinetics'
        self.data_dir                   = '/home/longteng/ssd/Kinetics400split'
        self.mode                       = 'train'

        self.num_sec_aud                = 1 # default
        self.aud_sample_rate            = 24000 # default
        self.aud_spec_type              = 2 
        self.use_volume_jittering       = False # default
        self.use_audio_temp_jittering   = False #default
        self.z_normalize                = True  # default for audio

        self.batch_size                 = 4
        self.workers                    = 0

        # model parameters
        self.vid_base_arch = 'r2plus1d_18'
        self.aud_base_arch = 'resnet9'
        self.norm_feat     = True
        self.use_mlp       = True
        self.headcount     = 10
        self.num_clusters  = 400

        # feature parameters
        self.weights_path  = Path('/home/longteng/ssd/selavi/weights/selavi_kinetics.pth') # should be str type
        self.a_layer_names = ['base.layer1', 'base.layer2', 'base.layer3', 'base.layer4', 'base.flatten']
        self.v_layer_names = ['base.layer4', 'base.flatten']
        self.feat_base     = Path('./feature/kinetics/train')

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
            import pdb
            pdb.set_trace()
            print("didnt load ", name)


if __name__ == '__main__':
    args = ARGS()
    data_module = K400DataModule(args.data_dir,
                                 batch_size = args.batch_size,
                                 num_workers = args.workers)
    # train_loader = data_module.train_dataloader()
    # valid_loader = data_module.val_dataloader()
    feat_train_loader = data_module.feat_train_dataloader()

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

    # clear cache at beginning
    torch.cuda.empty_cache()
    model.eval()

    model = model.cuda()

    feature_extractor_audio = create_feature_extractor(model.audio_network, return_nodes = args.a_layer_names)
    feature_extractor_video = create_feature_extractor(model.video_network, return_nodes = args.v_layer_names)

    feat_base = args.feat_base
    if not feat_base.exists():
        feat_base.mkdir()

    feat_levels = ['audio.' + item for item in args.a_layer_names] + ['video.' + item for item in args.v_layer_names]

    for batch in tqdm(feat_train_loader):

        frames, audio, label = batch['video'], batch['audio'], batch['label']
        video_name, video_index, clip_index = batch['video_name'], batch['video_index'], batch['clip_index']

        frames, audio = frames.cuda(), audio.cuda()

        v_feats_gpu = feature_extractor_video(frames)
        a_feats_gpu = feature_extractor_audio(audio)

        v_feats = {key: value.detach().cpu() for key, value in v_feats_gpu.items()}
        a_feats = {key: value.detach().cpu() for key, value in a_feats_gpu.items()}

        del v_feats_gpu, a_feats_gpu

        # Loop over each sample in batch
        for i in range(len(video_name)):

            # Define output path
            print(f'processing video {video_name[i]}, clip {clip_index[i]}')
            path_parts = Path(video_name[i]).parts
            class_name = path_parts[-2]
            file_name  = path_parts[-1]

            cls_base = feat_base / class_name

            if not cls_base.exists():
                cls_base.mkdir()

            out_path = (cls_base / file_name).with_suffix('.pth')
            out_feat_dict = dict()

            # Loop over each layer
            if out_path.exists():
                est_feat_dict_gpu = torch.load(out_path) # existing feature
                est_feat_dict = {key: value.detach().cpu() for key, value in est_feat_dict_gpu.items()}
                del est_feat_dict_gpu

                # process clip_index
                new_clip_index = clip_index[i].unsqueeze(dim = 0)
                est_clip_index = est_feat_dict['clip_index']
                out_feat_dict['clip_index'] = torch.hstack([est_clip_index, new_clip_index])

                # process features
                for layer_name in args.a_layer_names:

                    a_key = 'audio.' + layer_name
                    est_audio_layer_feat = est_feat_dict[a_key]
                    new_audio_layer_feat = a_feats[layer_name][i].unsqueeze(dim = 0)
                    out_feat_dict[a_key] = torch.vstack([est_audio_layer_feat, new_audio_layer_feat])

                for layer_name in args.v_layer_names:

                    v_key = 'video.' + layer_name
                    est_video_layer_feat = est_feat_dict[v_key]
                    new_video_layer_feat = v_feats[layer_name][i].unsqueeze(dim = 0)
                    out_feat_dict[v_key] = torch.vstack([est_video_layer_feat, new_video_layer_feat])

            else:
                # process clip_index
                out_feat_dict['clip_index'] = clip_index[i].unsqueeze(dim = 0)

                # process features
                for layer_name in args.a_layer_names:

                    a_key = 'audio.' + layer_name
                    out_feat_dict[a_key] = a_feats[layer_name][i].unsqueeze(dim = 0)

                for layer_name in args.v_layer_names:

                    v_key = 'video.' + layer_name
                    out_feat_dict[v_key] = v_feats[layer_name][i].unsqueeze(dim = 0)
            
            torch.save(out_feat_dict, out_path)
