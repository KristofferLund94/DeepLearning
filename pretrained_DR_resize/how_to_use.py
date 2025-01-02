import torch
import torch.nn as nn
from torchvision import models

if __name__ == '__main__':
    # You can use the code to replace the definition of 'self.backbone' in MyModel

    self.backbone = models.resnet18(pretrained=True)
    state_dict = torch.load('pretrained/resnet18.pth', map_location='cpu')
    info = self.backbone.load_state_dict(state_dict, strict=False)
    print('missing keys:', info[0])  # The missing fc or classifier layer is normal here
    print('unexpected keys:', info[1])

    # self.backbone = models.resnet34(pretrained=True)
    # state_dict = torch.load('pretrained/resnet34.pth', map_location='cpu')

    # self.backbone = models.vgg16(pretrained=True)
    # self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    # state_dict = torch.load('pretrained/vgg16.pth', map_location='cpu')

    # self.backbone = models.efficientnet_b0(pretrained=True)
    # state_dict = torch.load('pretrained/efficientnet_b0.pth', map_location='cpu')

    # self.backbone = models.densenet121(pretrained=True)
    # state_dict = torch.load('pretrained/densenet121.pth', map_location='cpu')

