import torch
import torch.nn as nn
from model.IR50 import Backbone as IR50
from model.ViT import ViT


class Network(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.dims = [64, 128, 256]
        self.VIT = ViT(dim=768, num_classes=self.num_classes, channels=147)

        self.ir_back = IR50()
        self.ir_back.load_state_dict(torch.load('./pretrained_models/ir50.pth'))

        self.conv1 = nn.Conv2d(in_channels=self.dims[0], out_channels=self.dims[0], kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.dims[1], out_channels=self.dims[1], kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.dims[2], out_channels=self.dims[2], kernel_size=3, stride=2, padding=1)

        self.embed1 = nn.Sequential(nn.Conv2d(self.dims[0], 768, kernel_size=3, stride=2, padding=1),
                                    nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1))
        self.embed2 = nn.Conv2d(self.dims[1], 768, kernel_size=3, stride=2, padding=1)
        self.embed3 = nn.Conv2d(self.dims[2], 768, kernel_size=1)

        self.avgp = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                  nn.Dropout(0.3))

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input, positive=None, negative=None, phase="test"):

        if phase == "train":
            anchor_1, anchor_2, anchor_3 = self.ir_back(input)
            positive_1, positive_2, positive_3 = self.ir_back(positive)
            negative_1, negative_2, negative_3 = self.ir_back(negative)
            # [N, 64, 56, 56], [N, 128, 28, 28], [N, 256, 14, 14]

            a1, a2, a3 = self.conv1(anchor_1), self.conv2(anchor_2), self.conv3(anchor_3)
            p1, p2, p3 = self.conv1(positive_1), self.conv2(positive_2), self.conv3(positive_3)
            n1, n2, n3 = self.conv1(negative_1), self.conv2(negative_2), self.conv3(negative_3)
            # [N, 64, 28, 28], [N, 128, 14, 14], [N, 256, 7, 7]

            a1, a2, a3 = self.embed1(a1), self.embed2(a2), self.embed3(a3)
            p1, p2, p3 = self.embed1(p1), self.embed2(p2), self.embed3(p3)
            n1, n2, n3 = self.embed1(n1), self.embed2(n2), self.embed3(n3)
            # [N, 768, 7, 7], [N, 768, 7, 7], [N, 768, 7, 7]

            o1, o2, o3 = a1.flatten(2).transpose(1, 2), a2.flatten(2).transpose(1, 2), a3.flatten(2).transpose(1, 2)
            # [N, 49, 768], [N, 49, 768], [N, 49, 768]

            o = torch.cat([o1, o2, o3], dim=1)
            # [N, 147, 768]

            o = self.VIT(o)
            out = self.softmax(o)

            a1, a2, a3 = self.avgp(a1), self.avgp(a2), self.avgp(a3)
            p1, p2, p3 = self.avgp(p1), self.avgp(p2), self.avgp(p3)
            n1, n2, n3 = self.avgp(n1), self.avgp(n2), self.avgp(n3)
            # [N, 768, 1, 1], [N, 768, 1, 1], [N, 768, 1, 1]

            a1, a2, a3 = torch.mean(a1, -1), torch.mean(a2, -1), torch.mean(a3, -1)
            p1, p2, p3 = torch.mean(p1, -1), torch.mean(p2, -1), torch.mean(p3, -1)
            n1, n2, n3 = torch.mean(n1, -1), torch.mean(n2, -1), torch.mean(n3, -1)
            # [N, 768, 1], [N, 768, 1], [N, 768, 1]

            a = torch.cat([a1, a2, a3], dim=2)
            p = torch.cat([p1, p2, p3], dim=2)
            n = torch.cat([n1, n2, n3], dim=2)
            # [N, 768, 3]

            return a, p, n, out

        else:
            anchor_1, anchor_2, anchor_3 = self.ir_back(input)

            a1, a2, a3 = self.conv1(anchor_1), self.conv2(anchor_2), self.conv3(anchor_3)

            o1, o2, o3 = self.embed1(a1).flatten(2).transpose(1, 2), self.embed2(a2).flatten(2).transpose(1, 2), \
                self.embed3(a3).flatten(2).transpose(1, 2)

            o = torch.cat([o1, o2, o3], dim=1)

            o = self.VIT(o)
            out = self.softmax(o)

            return out
