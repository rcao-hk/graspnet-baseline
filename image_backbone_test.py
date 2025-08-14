import os
# import sys
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

from models.pspnet import PSPNet
psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50')
}

import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)

class dino_extractor(nn.Module):
    def __init__(self, feat_ext):
        super(dino_extractor, self).__init__()

        if feat_ext == "dino":
            self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        else:
            raise NotImplementedError
        # self.transform = transform
        self.up_1 = Upsample(384, 256)
        self.up_2 = Upsample(256, 128)
        self.up_3 = Upsample(128, 64)
        
    def forward(self, img):
        # input_image = self.transform.apply_image_torch(img)
        B, _, H, W = img.size()
        features_dict = self.dino.forward_features(img)
        dino_feats = features_dict['x_norm_patchtokens'].view(B, H//14, W//14, -1)
        dino_feats = dino_feats.permute(0, 3, 1, 2)
        feat = self.up_1(dino_feats)
        feat = self.up_2(feat)
        feat = self.up_3(feat)
        feat = F.interpolate(feat, (H, W), mode='bilinear')
        return feat


print("Using PSPNet for feature extraction")
H, W = 224, 224
B = 4
image_input = torch.randn(B, 3, H, W).to(device)
img_extractor = PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=64, backend='resnet34').to(device)
img_extractor.eval()
emb = img_extractor(image_input)
print(emb.shape)

# feat_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
# feat_extractor.eval()
# features_dict = feat_extractor.forward_features(image_input)
# dino_feats = features_dict['x_norm_patchtokens'].view(B, H//14, W//14, -1)
# dino_feats = F.interpolate(dino_feats.permute(0, 3, 1, 2), (H, W), mode='bilinear')
print("Using DINO for feature extraction")
dino_img_extractor = dino_extractor("dino").to(device)
dino_img_extractor.eval()
dino_feats = dino_img_extractor(image_input)
print(dino_feats.shape)

print("Using segmentation_models_pytorch for ResNeXt feature extraction")
import segmentation_models_pytorch as smp
resnext_extractor = smp.Unet(encoder_name="resnext50_32x4d", encoder_weights="imagenet", in_channels=3, classes=64).to(device)
resnext_extractor.eval()
resnext_feats = resnext_extractor(image_input)

print(resnext_feats.shape)

for name, m in resnext_extractor.named_modules():
    print(name, m)
# for resnext_feat in resnext_feats:
#     print(resnext_feat.shape)