import torch
from torchvision import models as tmd
import timm
from .PC_model import pointnet2
from .localformer import TransformerEncoderLayer_CMA

class BenchMarkVQA(torch.nn.Module):
    def __init__(self,backbone):
        super(BenchMarkVQA,self).__init__()
        self.backbone = None
        if backbone == 'mc318':
            self.backbone = tmd.video.mc3_18(weights=tmd.video.MC3_18_Weights.DEFAULT)
        elif backbone == 'r3d18':
            self.backbone = tmd.video.r3d_18(weights=tmd.video.R3D_18_Weights.DEFAULT)
        elif backbone == 'r2p':
            self.backbone = tmd.video.r2plus1d_18(weights=tmd.video.R2Plus1D_18_Weights.DEFAULT)
        elif backbone == 'swin3d-b':
            self.backbone = tmd.video.swin3d_b(weights=tmd.video.Swin3D_B_Weights.DEFAULT)
        elif backbone == 'swin3d-s':
            self.backbone = tmd.video.swin3d_s(weights=tmd.video.Swin3D_S_Weights.DEFAULT)
        elif backbone == 'swin3d-t':
            self.backbone = tmd.video.swin3d_t(weights=tmd.video.Swin3D_T_Weights.DEFAULT)
        else:
            raise ValueError("Backbone not implement!")

        # for parm in self.backbone.parameters():
        #     parm.requires_grad = False
        self.linear = torch.nn.Linear(400,3)

    def forward(self,video):
        feature = self.backbone(video)
        out = self.linear(feature)
        return out[...,0], out[...,1], out[...,2]

class BenchMarkIQA(torch.nn.Module):
    def __init__(self,backbone):
        super(BenchMarkIQA, self).__init__()
        self.backbone = None
        if backbone == 'resnet18':
            self.backbone = timm.create_model('resnet18',pretrained=True,num_classes=3)
        elif backbone == 'resnet34':
            self.backbone = timm.create_model('resnet34',pretrained=True,num_classes=3)
        elif backbone == 'resnet50':
            self.backbone = timm.create_model('resnet50',pretrained=True,num_classes=3)
        elif backbone == 'vgg16':
            self.backbone = timm.create_model('vgg16',pretrained=True,num_classes=3)
        elif backbone == 'vgg19':
            self.backbone = timm.create_model('vgg19',pretrained=True,num_classes=3)
        elif backbone == 'swin-t':
            self.backbone = timm.create_model('swin_tiny_patch4_window7_224',pretrained=True,num_classes=3)
        elif backbone == 'swin-s':
            self.backbone = timm.create_model('swin_small_patch4_window7_224',pretrained=True,num_classes=3)
        elif backbone == 'swin-b':
            self.backbone = timm.create_model('swin_base_patch4_window7_224',pretrained=True,num_classes=3)
        elif backbone == 'swin-l':
            self.backbone = timm.create_model('swin_large_patch4_window7_224',pretrained=True,num_classes=3)
        elif backbone == 'cnniqa':
            from .cnniqa import CNNIQAnet
            self.backbone = CNNIQAnet()
        elif backbone == 'hyperiqa':
            from .hyperiqa import Hyperiqa
            self.backbone = Hyperiqa()
        elif backbone == 'maniqa':
            from .maniqa import MANIQA
            self.backbone = MANIQA(embed_dim=768,num_outputs=3)
        elif backbone == 'stairiqa':
            from .stairiqa import resnet50_imdt as stairiqa
            self.backbone = stairiqa()
        elif backbone == 'mobilenetv3':
            self.backbone = timm.create_model('mobilenetv3_small_100.lamb_in1k',pretrained=True,num_classes=3)
        elif backbone == 'convnext':
            self.backbone = timm.create_model('convnext_tiny.in12k_ft_in1k',pretrained=True,num_classes=3)
        elif backbone == 'hrnet':
            self.backbone = timm.create_model('hrnet_w18.ms_aug_in1k',pretrained=True,num_classes=3)
        else:
            raise ValueError("Backbone not implement!")

    def forward(self,img):
        out = self.backbone(img)
        return out[...,0], out[...,1], out[...,2]

'''
input_image : torch.Size([1, 3, 512, 512])
tensor3d : torch.Size([1, 1, 4096])
tensorco : torch.Size([1, 1, 4096])
transformed_feature : torch.Size([1, 4, 2304])
frames : torch.Size([1, 4, 3, 448, 448])
patches : torch.Size([1, 6, 3, 2048])
'''

class PatchFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(PatchFeatureExtractor, self).__init__()
        self.pc_backbone = pointnet2()

    def forward(self, patch):
        patch_size = patch.shape 
        patch = patch.view(-1,patch_size[2],patch_size[3])
        patch = self.pc_backbone(patch)
        patch = patch.view(patch_size[0],patch_size[1],-1)
        features = torch.mean(patch,dim=1)
        return features

class VideoFeatureExtractor(torch.nn.Module):
    def __init__(self, pretrained=pretrained):
        super(VideoFeatureExtractor, self).__init__()
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224.ms_in1k',pretrained=pretrained)

    def forward(self, video, video_features):
        b,f,c,h,w = video.shape 
        video = video.view(-1,c,h,w)
        vfs = video_features.shape 
        video_features = video_features.view(-1,vfs[-1])
        feat = self.backbone(video)
        feat = torch.cat((feat,video_features),dim=-1)
        feat = feat.view(b,f,-1)
        feat = torch.mean(feat,dim=1)
        return feat

class InputImageFeatureExtractor(torch.nn.Module):
    def __init__(self, pretrained=pretrained):
        super(InputImageFeatureExtractor, self).__init__()
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224.ms_in1k',pretrained=pretrained,num_classes=1024)

    def forward(self, image):
        feat = self.backbone(image)
        return feat

class TwoLayerLinear(torch.nn.Module):
    def __init__(self,in_ch,mi_ch,out_ch):
        super(TwoLayerLinear, self).__init__()
        self.regression_block = torch.nn.Sequential(
            torch.nn.Linear(in_ch,mi_ch),
            torch.nn.SiLU(),
            torch.nn.Linear(mi_ch,out_ch)
        )
    def forward(self,x):
        return self.regression_block(x)

class CMA_fusion(torch.nn.Module):
    def __init__(self,cma_planes=1024):
        super(CMA_fusion, self).__init__()
        self.encoder = TransformerEncoderLayer_CMA(d_model=cma_planes,nhead=8, dim_feedforward = 2048, dropout = 0.1)

    def forward(self,A, B):
        A_a,B_a = self.encoder(A,B)
        return A_a+B_a

class MultiFeatureModel(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(MultiFeatureModel, self).__init__()
        self.patch_extractor = PatchFeatureExtractor()
        self.video_extractor = VideoFeatureExtractor(pretrained=pretrained)
        self.image_extractor = InputImageFeatureExtractor(pretrained=pretrained)
        self.videof_remap = TwoLayerLinear(3304, 2048, 1024)
        self.patchf_remap = TwoLayerLinear(1024, 1024, 1024)
        self.lmmf_remap = TwoLayerLinear(4096*2, 1024, 1024)
        self.cma_iv = CMA_fusion()
        self.cma_ip = CMA_fusion()
        self.cma_il = CMA_fusion()
        self.final_regresstion = torch.nn.Sequential(
            TwoLayerLinear(1024*7,1024*2,1024),
            torch.nn.SiLU(),
            TwoLayerLinear(1024,512,3)
        )

    def forward(self,input_image, 
                frames,
                transformed_feature,
                patches,
                tensor3d,
                tensorco):
        image_feature = self.image_extractor(input_image) # 1024

        video_feature = self.video_extractor(frames, transformed_feature) # 3304
        video_feature = self.videof_remap(video_feature)

        patch_feature = self.patch_extractor(patches) # 1024
        patch_feature = self.patchf_remap(patch_feature)
        tensor3d, tensorco = tensor3d.to(torch.float32), tensorco.to(torch.float32)
        tensor3d, tensorco = tensor3d.squeeze(1), tensorco.squeeze(1)
        lmm_feature = self.lmmf_remap(torch.cat((tensor3d,tensorco),dim=-1))

        iv_feature = self.cma_iv(image_feature,video_feature)
        ip_feature = self.cma_ip(image_feature,patch_feature)
        il_feature = self.cma_il(image_feature,lmm_feature)

        score = self.final_regresstion(
            torch.cat((image_feature,video_feature,patch_feature,lmm_feature,
                       iv_feature,ip_feature,il_feature),dim=-1)
        )

        # return score
        return score[...,0], score[...,1], score[...,2]
