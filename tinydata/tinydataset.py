import numpy as np
from fontTools.misc.psOperators import ps_string
from torch.utils.data import Dataset
from typing import Tuple, List
import os
import cv2
from utils.util import read_lables_file, xyxy2xywh
import torch

class TinyData(Dataset):
    def __init__(self, data_dir: str=None, img_size: int=640):
        self.data_dir = data_dir
        self.data: List[Tuple[str, str]] = [(os.path.join(self.data_dir[0], item), os.path.join(self.data_dir[-1], item.replace(".jpg", ".txt"))) for item in os.listdir(self.data_dir[0])]
        self.images_size = img_size

    @property
    def data_dir(self) -> Tuple[str, str]:
        return self.__image_dir, self.__label_dir

    @data_dir.setter
    def data_dir(self, value: str) -> None:
        if os.path.isdir(value):
            for folder in os.listdir(value):
                if folder.split('/')[-1] == 'images':
                    self.__image_dir = os.path.join(value, 'images')
                elif folder.split('/')[-1] == "labels":
                    self.__label_dir = os.path.join(value, 'labels')
                else:
                    continue

    def __load_image(self, index: int) -> Tuple[str, np.ndarray, Tuple[int, int], Tuple[int, int]]:
        path, lbl_path = self.data[index]

        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.images_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return lbl_path, img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

    @staticmethod
    def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        ratio = r, r

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, item: int):
        lbl_path, img, (h0, w0), (h, w) = self.__load_image(item)
        shape = self.images_size
        img, ratio, pad = TinyData.letterbox(img, shape, auto=False, scaleup=False)

        shapes = (h0, w0) , ((h / h0, w / w0), pad)

        labels = []
        x = read_lables_file(lbl=lbl_path)
        if x.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = x.copy()
            labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_out, self.data[item][0], shapes

    @staticmethod
    def collate_fn(batch):
        #img, ir, label, path, shapes = zip(*batch)  # transposed  #zjq
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        #return torch.stack(img, 0), torch.stack(ir, 0), torch.cat(label, 0), path, shapes #zjq
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    def __repr__(self):
        return f"TinyData(data_dir={self.data_dir})"
