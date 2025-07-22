import sys
import torch
import torch.nn as nn
from torch import Tensor
from tinynetwork.supreres import SuperResolution
from ultralytics import YOLO
from typing import List
import torch.functional as F

YOLO_MODEL = YOLO("yolo12n.pt").model.model


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        try:
            out = torch.cat(x, self.d)
            return out
        except Exception as error:
            print(f"error '{error}'")
            print(f"tensor1: {x[0].shape} | tensor2: {x[-1].shape}")
            b0, c0, w0, h0 = x[0].shape
            b1, c1, w1, h1 = x[-1].shape
            if w0 == w1 and h0 != h1:
                pad = torch.rand(b0, c0, w0, h0 - h1)
                x[-1] = torch.cat((x[-1], pad), dim=3)
            elif w0 != w1 and h0 != h1:
                x[-1] = F.interpolate(x[-1], size=(w0, h0), mode='bilinear', align_corners=False)
            out = torch.cat(x, self.d)
            return out

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
        return p3, p4, p5

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
    def __init__(self, yolo_model_src: str) -> None:
        super().__init__()
        yolo_model = YOLO(yolo_model_src).model.model
        self.backbone = yolo_model[:9]
        self.neck = yolo_model[9:21]
        self.head = yolo_model[21]
        self.super_res = SuperResolution(num_channel=3, c1=64, c2=256, scale_factor=2)
        self.storage_layer_output: List[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.backbone):
            try:
                x = layer(x)
                if i == 2 or i == 7 or i == 8:
                    self.storage_layer_output.append(x)
            except Exception as error:
                print(i, f"error: {error}")
        output_res = self.super_res(self.storage_layer_output[1], self.storage_layer_output[0])
        output_neck = self.neck(self.storage_layer_output[-1])
        output = self.head(output_neck)
        return self.storage_layer_output, output

