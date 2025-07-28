import torch
import torch.nn as nn
from torch import Tensor
from tinynetwork.supreres import SuperResolution
from ultralytics import YOLO
from utils.util import active_trainable_params
from typing import Tuple

YOLO_MODEL = YOLO("/home/my_proj/weights/pretrained_models/yolo12n.pt").model.model
active_trainable_params(YOLO_MODEL)

class BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = YOLO_MODEL[0]
        self.l1 = YOLO_MODEL[1]
        self.l2 = YOLO_MODEL[2]
        self.l3 = YOLO_MODEL[3]
        self.l4 = YOLO_MODEL[4]
        self.l5 = YOLO_MODEL[5]
        self.l6 = YOLO_MODEL[6]
        self.l7 = YOLO_MODEL[7]
        self.l8 = YOLO_MODEL[8]

    def forward(self, x):
        p1 = self.l0(x)
        p2 = self.l1(p1)
        p2 = self.l2(p2)
        p3 = self.l3(p2)
        p3 = self.l4(p3)
        p4 = self.l5(p3)
        p4 = self.l6(p4)
        p5 = self.l7(p4)
        p5 = self.l8(p5)
        return p2, p3, p4, p5

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.l9 = YOLO_MODEL[9]
        self.l10 = YOLO_MODEL[10]
        self.l11 = YOLO_MODEL[11]
        self.l12 = YOLO_MODEL[12]
        self.l13 = YOLO_MODEL[13]
        self.l14 = YOLO_MODEL[14]
        self.l15 = YOLO_MODEL[15]
        self.l16 = YOLO_MODEL[16]
        self.l17 = YOLO_MODEL[17]
        self.l18 = YOLO_MODEL[18]
        self.l19 = YOLO_MODEL[19]
        self.l20 = YOLO_MODEL[20]
        self.l21 = YOLO_MODEL[21]
        for idx, module in enumerate(self.l21.cv3):
            for jdx, block in enumerate(module):
                if jdx == 2:
                    block.out_channels = 1
        # for idx, module in enumerate(self.l21.cv3):
        #     for jdx, block in enumerate(module):
        #         if jdx == 2:
        #             block.in_channels = 18
        #             block.out_channels = 18
        #             break
        #         for kdx, layers in enumerate(block):
        #             if idx == 0 and jdx == 0 and kdx == 0:
        #                 continue
        #             elif idx == 0 and jdx == 0 and kdx == 1:
        #                 layers.conv.out_channels = 18
        #                 layers.bn.num_features = 18
        #             else:
        #                 layers.conv.in_channels = 18
        #                 layers.conv.out_channels = 18
        #                 layers.bn.num_features = 18


        # for item in self.l21.cv3:
        #     for idx, layer in enumerate(item):
        #         if idx == 2:
        #             break
        #         for idx, l in enumerate(layer):
        #             if idx == 1:
        #                 l.conv.out_channels = 1
        #                 l.bn.num_features=1

    def forward(self, p3, p4, p5):
        u1 = self.l9(p5)
        c1 = self.l10((u1, p4))
        h1 = self.l11(c1)
        u2 = self.l12(h1)
        c2 = self.l13((u2, p3))
        h2 = self.l14(c2)
        h3 = self.l15(h2)
        c3 = self.l16((h3, p4))
        h4 = self.l17(c3)
        h5 = self.l18(h4)
        c4 = self.l19((h5, p5))
        h6 = self.l20(c4)
        d = self.l21([h2, h4, h6])
        return d

class SuperYoloo(nn.Module):
    def __init__(self, tr_model: bool=True):
        super().__init__()
        self.tr_mode = tr_model
        self.back_bone = BackBone()
        self.super = SuperResolution(num_channel=3, c1=64, c2=128, scale_factor=2)
        self.head = Head()

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        if self.tr_mode:
            p2, p3, p4, p5 = self.back_bone(input)
            output = self.head(p3, p4, p5)
            super_output = self.super(p3, p2)
            return output, super_output
        else:
            p2, p3, p4, p5 = self.back_bone(input)
            output = self.head(p3, p4, p5)
            return output
