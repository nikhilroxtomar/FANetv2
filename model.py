import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pvtv2 import pvt_v2_b2, pvt_v2_b3
from text2embed import Text2Embed
import numpy as np
import cv2

def save_feats_mean(x, size=(256, 256)):
    b, c, h, w = x.shape
    with torch.no_grad():
        x = x.detach().cpu().numpy()
        x = np.transpose(x[0], (1, 2, 0))
        x = np.mean(x, axis=-1)
        x = x/np.max(x)
        x = x * 255.0
        x = x.astype(np.uint8)

        if h != size[1]:
            x = cv2.resize(x, size)

        x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
        x = np.array(x, dtype=np.uint8)
        return x

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, act=True):
        super().__init__()

        layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_c),
        )
        if act == True: layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.conv(inputs)

class dilated_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.c1 = nn.Sequential(conv_block(in_c, out_c, kernel_size=1, padding=0))
        self.c2 = nn.Sequential(conv_block(in_c, out_c, kernel_size=(3, 3), padding=6, dilation=6))
        self.c3 = nn.Sequential(conv_block(in_c, out_c, kernel_size=(3, 3), padding=12, dilation=12))
        self.c4 = nn.Sequential(conv_block(in_c, out_c, kernel_size=(3, 3), padding=18, dilation=18))
        self.c5 = conv_block(out_c*4, out_c, kernel_size=3, padding=1, act=False)
        self.c6 = conv_block(in_c, out_c, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        xc = torch.cat([x1, x2, x3, x4], axis=1)
        xc = self.c5(xc)
        xs = self.c6(x)
        x = self.relu(xc+xs)
        return x

class feature_enhancement_dilated_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.pool = nn.MaxPool2d((2, 2))
        self.c1 = conv_block(in_c[0], out_c)
        self.c2 = conv_block(out_c+in_c[1], out_c)
        self.c3 = conv_block(out_c+in_c[2], out_c)
        self.c4 = conv_block(out_c+in_c[3], out_c)
        self.dc = dilated_conv(out_c, out_c)

    def forward(self, f1, f2, f3, f4):
        # print(f1.shape, f2.shape, f3.shape, f4.shape)
        x = self.pool(self.c1(f1))
        x = torch.cat([x, f2], dim=1)

        x = self.pool(self.c2(x))
        x = torch.cat([x, f3], dim=1)

        x = self.pool(self.c3(x))
        x = torch.cat([x, f4], dim=1)

        x = self.c4(x)
        x = self.dc(x)

        return x

class image_textual_feature_fusion_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.l1 = nn.Sequential(nn.Linear(in_c[1], out_c), nn.ReLU(inplace=True))
        self.l2 = nn.Sequential(nn.Linear(8*8+16, 16*16), nn.ReLU(inplace=True))
        self.c1 = conv_block(256, out_c)

    def forward(self, image_feats, text_feats):
        txt_f = self.l1(text_feats).transpose(1, 2)
        img_f = image_feats.view((image_feats.shape[0], image_feats.shape[1], image_feats.shape[2]*image_feats.shape[3]))
        combine = torch.cat([txt_f, img_f], dim=2)
        combine = self.l2(combine)
        combine = combine.view((combine.shape[0], combine.shape[1], 16, 16))
        combine = self.c1(combine)
        return combine

class mask_attention_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = conv_block(in_c, out_c)
        self.c2 = conv_block(in_c, out_c)
        self.c3 = conv_block(out_c*2, out_c, kernel_size=1, padding=0)

    def forward(self, x, mask):
        """ Mask processing """
        mask = F.interpolate(mask, size=(x.shape[2:]))
        mask = torch.sigmoid(mask)

        """ Spatial Attention """
        xf = self.c1(x * mask)
        x2 = self.c2(x)
        x3 = torch.cat([xf, x2], dim=1)
        x4 = self.c3(x3)

        return x4

def image_encoder(image_size, checkpoint_path):
    encoder = pvt_v2_b3()  ## [64, 128, 320, 512]
    save_model = torch.load(checkpoint_path)
    model_dict = encoder.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    encoder.load_state_dict(model_dict)
    return encoder

class Model(nn.Module):
    def __init__(self,
        checkpoint_path="/media/nikhil/LAB/ML/ME/2023/pretrained/PVT/pvt_v2_b3.pth",
        image_size=256,
        num_classes=1,
        max_length=16
        ):
        super().__init__()

        """ Encoder """
        self.encoder = image_encoder(image_size=image_size, checkpoint_path=checkpoint_path)
        self.encoder_feats_fusion = feature_enhancement_dilated_block([64, 128, 320, 512], 512)

        """ Classification """
        self.avg_pool = nn.AdaptiveAvgPool2d((1))
        self.num_polyps = nn.Linear(512, 1)
        self.size_small = nn.Linear(512, 1)
        self.size_medium = nn.Linear(512, 1)
        self.size_large = nn.Linear(512, 1)

        """ Bottleneck """
        self.dc = dilated_conv(512, 512)
        self.c0 = conv_block(512, 256)
        self.embedding = Text2Embed(max_length=max_length)
        self.ff = image_textual_feature_fusion_block([256, 300], 256)

        """ Decoder """
        self.c1 = nn.Sequential(conv_block(256+320, 256), conv_block(256, 256))
        self.m1 = mask_attention_block(256, 256)

        self.c2 = nn.Sequential(conv_block(256+128, 128), conv_block(128, 128))
        self.m2 = mask_attention_block(128, 128)

        self.c3 = nn.Sequential(conv_block(128+64, 64), conv_block(64, 64))
        self.m3 = mask_attention_block(64, 64)

        self.c4 = nn.Sequential(conv_block(64+3, 32), conv_block(32, 32))
        self.m4 = mask_attention_block(32, 32)

        """ Output """
        self.mask_output = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, image, mask, heatmap=False):
        """ Encoder """
        f1, f2, f3, f4 = self.encoder(image)
        # print(f1.shape, f2.shape, f3.shape, f4.shape)

        f5 = self.encoder_feats_fusion(f1, f2, f3, f4)

        """ Classification """
        f5_pool = self.avg_pool(f5).view(f5.shape[0], f5.shape[1])

        num_polyps = self.num_polyps(f5_pool)
        size_small = self.size_small(f5_pool)
        size_medium = self.size_medium(f5_pool)
        size_large = self.size_large(f5_pool)

        """ Bottleneck """
        down_mask = F.interpolate(mask, size=(f5.shape[2], f5.shape[3]), mode="bilinear", align_corners=True)
        x = f5 + down_mask
        x = self.c0(x)

        text_prompts = self.to_prompt(num_polyps, [size_small, size_medium, size_large])
        x = self.ff(x, text_prompts)

        """ Decoder """
        x = torch.cat([x, f3], dim=1)
        x = self.c1(x)
        x = self.m1(x, mask)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        x = torch.cat([x, f2], dim=1)
        x = self.c2(x)
        x = self.m2(x, mask)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        x = torch.cat([x, f1], dim=1)
        x = self.c3(x)
        x = self.m3(x, mask)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)

        x = torch.cat([x, image], dim=1)
        x = self.c4(x)
        h1 = save_feats_mean(x)
        x = self.m4(x, mask)
        h2 = save_feats_mean(x)

        """ Output """
        y = self.mask_output(x)

        if heatmap == True:
            return [num_polyps, size_small, size_medium, size_large], y, [h1, h2]
        else:
            return [num_polyps, size_small, size_medium, size_large], y

    def to_prompt(self, num, size):
        small, medium, large = size

        text_prompts = []

        pred_num = torch.sigmoid(num)
        pred_small = torch.sigmoid(small)
        pred_medium = torch.sigmoid(medium)
        pred_large = torch.sigmoid(large)

        for i in range(num.shape[0]):
            prompt = "a colorectal image with "
            pn = (pred_num[i] > 0.5).int()

            if pn == 0: prompt += "one "
            else: prompt += "many "

            ps = (pred_small[i] > 0.5).int()
            pm = (pred_medium[i] > 0.5).int()
            pl = (pred_large[i] > 0.5).int()

            if ps == 1: prompt += "small "
            if pm == 1: prompt += "medium "
            if pl == 1: prompt += "large "

            if pn == 0: prompt += "sized polyp"
            elif pn == 1: prompt += "sized polyps"

            embed = prompt_embedding = self.embedding.to_embed(prompt)
            text_prompts.append(embed)

        text_prompts = np.array(text_prompts, np.float32)
        text_prompts = torch.from_numpy(text_prompts)
        device = num.get_device()
        if device == -1:
            text_prompts = text_prompts.to("cpu")
        else:
            text_prompts = text_prompts.to("cuda")

        return text_prompts

if __name__ == "__main__":
    image = torch.randn((8, 3, 256, 256))
    mask = torch.randn((8, 1, 256, 256))

    model = Model()
    [num_polyps, size_small, size_medium, size_large], y = model(image, mask)
    print(num_polyps.shape, size_small.shape, size_medium.shape, size_large.shape, y.shape)
