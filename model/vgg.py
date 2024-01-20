import os

import torch
import torch.nn as nn

import torchvision


class VGG16FeatureExtractor(nn.Module):

    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()

        vgg16 = torchvision.models.vgg16(pretrained=True)

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, images):
        results = [images]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

# ----------------------------------------
#            Perceptual Network
# ----------------------------------------
# VGG-16 conv4_3 features
class PerceptualNetC(nn.Module):
    def __init__(self):
        super(PerceptualNetC, self).__init__()
        os.environ['TORCH_HOME'] = './data/'
        vgg16 = torchvision.models.vgg16(pretrained=False)
        pre = torch.load(r'/opt/data/private/gaoyanjun/CMP_ALL/data/vgg16-397923af.pth')
        vgg16.load_state_dict(pre)
        self.conv0 = nn.Conv2d(18, 3, kernel_size=3, stride=1, padding=1,bias=False)
        block = [vgg16.features[:15].eval()]
        for p in block[0]:
            p.requires_grad = False
        self.block = torch.nn.ModuleList(block)
        self.transform = torch.nn.functional.interpolate
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        x = (x-self.mean) / self.std
        x = x.view(x.shape[0],-1,x.shape[3],x.shape[4])
        x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
        x = self.conv0(x)
        for block in self.block:
            x = block(x)
        return x

class PerceptualNetF(nn.Module):
    def __init__(self):
        super(PerceptualNetF, self).__init__()
        os.environ['TORCH_HOME'] = './data/'
        vgg16 = torchvision.models.vgg16(pretrained=False)
        pre = torch.load(r'/opt/data/private/gaoyanjun/CMP_ALL/data/vgg16-397923af.pth')
        vgg16.load_state_dict(pre)
        self.conv0 = nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1,bias=False)
        block = [vgg16.features[:15].eval()]
        for p in block[0]:
            p.requires_grad = False
        self.block = torch.nn.ModuleList(block)
        self.transform = torch.nn.functional.interpolate
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        x = (x-self.mean) / self.std
        x = x.view(x.shape[0],-1,x.shape[3],x.shape[4])
        x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
        x = self.conv0(x)
        for block in self.block:
            x = block(x)
        return x