import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from .criterion.relative_depth import relative_depth_crit


class ResidualConv(nn.Module):
    def __init__(self, in_channel):
        super(ResidualConv, self).__init__()
        self.trans = nn.Conv2d(in_channel, 256, 3, padding=1)
        self.conv = nn.Sequential(
                        nn.ReLU(True),
                        nn.Conv2d(256, 256, 3, padding=1),
                        nn.ReLU(True),
                        nn.Conv2d(256, 256, 3, padding=1)
                    )

    def forward(self, x):
        x = self.trans(x)
        return self.conv(x) + x


class ConvUpsampling(nn.Module):
    def __init__(self, in_channel):
        super(ConvUpsampling, self).__init__()
        self.convup = ResidualConv(in_channel)

    def forward(self, x):
       return F.upsample(self.convup(x), scale_factor=2, mode="bilinear")


class FeatureFusion(nn.Module):
    def __init__(self, input1):
        super(FeatureFusion, self).__init__()
        self.left = ResidualConv(input1)
        self.convup = ConvUpsampling(256)

    def forward(self, left_x, top_x):
        return self.convup(self.left(left_x) + top_x)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # torchvision model
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        # use last batch normalization weight size as output dim. (Not elegant)
        out_dim4 = list(self.layer4.modules())[-2].weight.size()[0]
        out_dim3 = list(self.layer3.modules())[-2].weight.size()[0]
        out_dim2 = list(self.layer2.modules())[-2].weight.size()[0]
        out_dim1 = list(self.layer1.modules())[-2].weight.size()[0]
        self.up4 = ConvUpsampling(out_dim4)
        # fusion modules
        self.fu1 = FeatureFusion(out_dim1)
        self.fu2 = FeatureFusion(out_dim2)
        self.fu3 = FeatureFusion(out_dim3)
        # adaptive output layer
        self.adapt = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.Conv2d(128, 1, 3, padding=1),
        )

    def forward(self, x):
        # forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        # fusion and upsampling
        x = self.up4(out4)
        x = self.fu3(out3, x)
        x = self.fu2(out2, x)
        x = self.fu1(out1, x)
        return F.upsample(self.adapt(x), scale_factor=2, mode="bilinear")


def get_model():
    return Model().cuda()


def get_criterion():
    return relative_depth_crit(ranking=True)

        
if __name__ == '__main__':
    model = Model().cuda()
    inputs = torch.zeros((1, 3, 224, 224), device=torch.device("cuda:0"))
    model.train(True)
    output = model(inputs)
    print(output)
