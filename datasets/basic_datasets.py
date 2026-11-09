import os 
import torch, torchvision
import numpy as np
from einops import rearrange
from PIL import Image
from torchvision import transforms

import decord
decord.bridge.set_bridge('torch')

class I23DCQA_VideoDataset(torch.utils.data.Dataset):
    def __init__(self,root,mode,ann_file,sample_num,size):
        super(I23DCQA_VideoDataset).__init__()
        self.root = root
        self.mode = mode
        self.ann_file = ann_file
        self.sample_num = sample_num
        self.size = size
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        if isinstance(self.ann_file, list):
            self.videos_infos = self.ann_file
        else:
            raise ValueError('ann_file must a list, not support other format!')

    def _sample_frames(self):
        if self.mode != 'train':
            t_vals = np.linspace(0., 0.999, num= self.sample_num)
            return t_vals
        else:
            t_vals = np.linspace(0., 1., num= self.sample_num)
            mids = 0.5 * ( t_vals[:-1] + t_vals[1:] )
            lowers = np.concatenate([t_vals[:1],mids],axis=-1)
            uppers = np.concatenate([mids,t_vals[-1:]],axis=-1)
            rand_vals = np.random.rand(self.sample_num)
            samples = lowers + (uppers - lowers) * rand_vals
            return samples

    def __len__(self):
        return len(self.videos_infos)

    def __getitem__(self,index):
        video_info = self.videos_infos[index]
        video_path = video_info['video_path']
        overall = video_info['overall'] / 100
        color = video_info['color'] / 100
        shape = video_info['shape'] / 100
        
        vreader = decord.VideoReader(os.path.join(self.root,video_path))
        samples = np.floor(self._sample_frames() * len(vreader)).astype(np.int64)
        imgs = [vreader[idx] for idx in samples]
        video = torch.stack(imgs,dim=0)
        video = rearrange(video,'t w h c -> c t w h')
        video = torch.nn.functional.interpolate(video,size=(self.size,self.size))
        video = ((video.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)

        data = {
            'video': video,
            'overall': overall,
            'color': color,
            'shape': shape
        }
        return data

# Feature directory: features/<category>/<method>/<object>/
# /features/class/methods/inst/
#                             --3D.pt
#                             --compare.pt
#                             --patch.npy
#                             --feature_{1-4}_{slow/fast}_feature.npy
#                             --frames 
#                                     -00{0-3}.png
class I23DCQA_FeatureDataset(torch.utils.data.Dataset):
    def __init__(self,root,
                 mode,
                 ann_file,
                 transform,
                 video_height_crop=224,
                 video_width_crop=224,
                 vf_type='SlowFast',
                 ):
        super(I23DCQA_FeatureDataset).__init__()
        self.root = root 
        self.mode = mode
        self.ann_file = ann_file 

        self.transform = transform
        self.imagets = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
        self.video_height_crop = video_height_crop
        self.video_width_crop = video_width_crop
        self.vf_type = vf_type
        if isinstance(self.ann_file, list):
            self.videos_infos = self.ann_file
        else:
            raise ValueError('ann_file must a list, not support other format!')


    def __len__(self):
        return len(self.videos_infos) 

    def __read_patches(self,patch_path,patch_length_read=6,npoint=2048):
        selected_patches = torch.zeros([patch_length_read, 3, npoint])
        points = list(np.load(patch_path))
        # randomly select patches during the training stage
        if len(points) < patch_length_read:
            points = random.choices(points,k=patch_length_read)
        if self.mode=='train':
            random_patches = random.sample(points, patch_length_read)
        else:
            random_patches = points
        for i in range(patch_length_read):
            selected_patches[i] = torch.from_numpy(random_patches[i]).transpose(0,1)
        return selected_patches
    def __get_frames(self,frames_dir):
        transformed_video = torch.zeros([8, 3, self.video_height_crop, self.video_width_crop])
        for i in range(8):
            path = os.path.join(frames_dir,'{:03d}'.format(i) + '.png')
            read_frame = Image.open(path)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame
        return transformed_video

    def __get_videofeatures(self,features_dir):
        feature_lenth = 8
        if self.vf_type=='Slow':
            transformed_feature = torch.zeros([feature_lenth, 2048])
            for i in range(feature_lenth):
                feature_3D = np.load(os.path.join(features_dir, 'feature_' + str(i) + '_slow_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.vf_type=='Fast':
            transformed_feature = torch.zeros([feature_lenth, 256])
            for i in range(feature_lenth):
                feature_3D = np.load(os.path.join(features_dir, 'feature_' + str(i) + '_fast_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        else:
            transformed_feature = torch.zeros([feature_lenth, 2048+256])
            for i in range(feature_lenth):
                feature_3D_slow = np.load(os.path.join(features_dir, 'feature_' + str(i) + '_slow_feature.npy'))
                feature_3D_fast = np.load(os.path.join(features_dir, 'feature_' + str(i) + '_fast_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D
        return transformed_feature

    def __get_lmmfeatures(self,features_dir):
        tensor3d_path = os.path.join(features_dir,'3D.pt')
        tensorco_path = os.path.join(features_dir,'compare2.pt')
        tensor3d = torch.load(tensor3d_path,map_location=torch.device('cpu'),weights_only=True)
        tensorco = torch.load(tensorco_path,map_location=torch.device('cpu'),weights_only=True)
        return tensor3d, tensorco


    def __getitem__(self, index):
        video_info = self.videos_infos[index]
        image_path = video_info['image_path']
        video_path = video_info['video_path']
        overall = video_info['overall'] / 100
        color = video_info['color'] / 100
        shape = video_info['shape'] / 100

        input_image = Image.open(os.path.join(self.root,image_path))
        input_image = input_image.convert('RGB')
        input_image = self.imagets(input_image)
        features_path = os.path.join(self.root,video_path.replace('video_single','features').replace('.mp4',''))
        tensor3d, tensorco = self.__get_lmmfeatures(features_path)
        transformed_feature = self.__get_videofeatures(features_path)
        frames = self.__get_frames(os.path.join(features_path,'frames'))
        patches = self.__read_patches(os.path.join(features_path,'patch.npy'))

        data = {
            'input_image':input_image,
            'tensor3d':tensor3d,
            'tensorco':tensorco,
            'transformed_feature':transformed_feature,
            'frames':frames,
            'patches':patches,
            'overall':overall,
            'shape':shape,
            'color':color,
        }
        return data


import random
def train_test_split(dataset_path, ann_file, ratio=0.8, seed=42):
    random.seed(seed)
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split("|")
            image_path, mesh_path, video_path, overall, color, shape = line_split
            overall = float(overall)
            color = float(color)
            shape = float(shape)
            # filename = os.path.join(dataset_path, filename)
            video_infos.append(dict(image_path=image_path,
                                    mesh_path=mesh_path,
                                    video_path=video_path,
                                    overall=overall,
                                    color=color,
                                    shape=shape))
    random.shuffle(video_infos)
    return (
        video_infos[: int(ratio * len(video_infos))],
        video_infos[int(ratio * len(video_infos)) :],
    )
