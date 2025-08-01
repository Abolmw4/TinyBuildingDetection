import unittest

import torch
from scipy.constants import troy_ounce


class MyTestCase(unittest.TestCase):
    def test_supernet_input_output(self):
        from tinynetwork.supreres import SuperResolution
        import torch

        superNet = SuperResolution(num_channel=3, c1=128, c2=512, scale_factor=2).to('cuda')
        low_level_feat = torch.rand((1, 128, 64, 64)).to('cuda')
        high_level_feat = torch.rand((1, 512, 16, 16)).to('cuda')

        output_res = superNet(high_level_feat, low_level_feat)
        print(output_res.shape)

    def test_supernet_info(self):
        from tinynetwork.supreres import SuperResolution
        from utils.util import count_parameters, count_trainable_parameters, count_layer_wise_parameters

        superNet = SuperResolution(num_channel=3, c1=128, c2=512, scale_factor=2).to('cuda')

        print(f"All parameter: {count_parameters(superNet)}")

        print(f"trainable parameters: {count_trainable_parameters(superNet)}")

        print(f"layer wise parameters: {count_layer_wise_parameters(superNet)}")

    def test_backbone_to_superres(self):
        import torch
        from tinynetwork.supreres import SuperResolution
        import torch.nn as nn
        import torch.optim as optim

        l2_yolo_back_bone = torch.rand(1, 64, 160, 160)
        l4_yolo_back_bone = torch.rand(1, 128, 80, 80)
        l7_yolo_back_bone = torch.rand(1, 256, 20, 20)

        superNet = SuperResolution(num_channel=3, c1=64, c2=128, scale_factor=2)
        output_res = superNet(l4_yolo_back_bone, l2_yolo_back_bone)
        print(output_res.shape)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(superNet.parameters(), lr=0.01)
        target = torch.rand(1, 3, 1280, 1280)

        loss = criterion(output_res, target)
        optimizer.zero_grad()
        loss.backward()
        for name, param in superNet.named_parameters():
            if param.grad is None:
                print("\U0001F468", f"No gradient for {name}")
                return False
            if torch.isnan(param.grad).any():
                print("\U0001F648", f"NaN in gradient of {name}")
                return False
            if torch.all(param.grad == 0):
                print(f"\U0001F468", f"All-zero gradient for {name}")

                return False
            print(f"U+1F648  Gradient OK for {name}")
        return True


    def test_load_and_getinfo_yolov12n(self):
        from ultralytics import YOLO
        from utils.util import count_parameters, count_trainable_parameters, count_layer_wise_parameters
        import torch

        model = YOLO("/home/my_proj/weights/pretrained_models/yolo12n.pt")
        # x = torch.rand(1, 3, 640, 640)
        # out = model(x)
        # print(out)
        print(model.model.model[21].cv2[-1][-1])
        # print(model)
        # print(f"All parameter: {count_parameters(model)}")
        #
        # print(f"trainable parameters: {count_trainable_parameters(model)}")
        #
        # print(f"layer wise parameters: {count_layer_wise_parameters(model)}")

    def test_get_backbone_neck_head_yolov12n(self):
        from ultralytics import YOLO
        import torch

        yolo_model = YOLO("/home/my_proj/weights/pretrained_models/yolo12n.pt")

        full_model = yolo_model.model.model
        back_bone = full_model[:9]
        neck = full_model[9:21]
        head = full_model[21]

        print(back_bone[6])

    def test_get_output_yolo_model_layer(self):
        from ultralytics import YOLO
        import torch

        yolo_model = YOLO("/home/my_proj/weights/pretrained_models/yolo12n.pt")
        dummy_input = torch.rand(1, 3, 640, 640)
        output = yolo_model(dummy_input)
        print(output)

    def test_get_backbone_output_layers(self):
        from ultralytics import YOLO
        import torch

        yolo_model = YOLO("/home/my_proj/weights/pretrained_models/yolo12n.pt")
        backbone = yolo_model.model.model[:9]  # Sequential([...])
        dummy_input = torch.rand(1, 3, 640, 640)

        x = dummy_input
        for i, layer in enumerate(backbone):
            try:
                x = layer(x)
                print(f"Layer {i} output shape: {x.shape}")
            except Exception as error:
                print(f"error: {error}")

    def test_get_ouput_from_yolo_model(self):
        from tinynetwork.yoloo import SuperYoloo
        import torch

        model = SuperYoloo()
        print(model)
        import sys; sys.exit()
        input = torch.rand(1, 3, 640, 640)
        output, _ = model(input)


    def test_custome_yolo(self):
        from tinynetwork.yoloo import BackBone
        import torch

        backbone_net = BackBone()
        input = torch.rand(1, 3, 640, 640)

        p3, p4, p5 = backbone_net(input)
        print(p5)

    def test_custome_yolo_head(self):
        from tinynetwork.yoloo import BackBone, Head
        import torch

        backbone_net = BackBone()
        head_net = Head()
        input = torch.rand(1, 3, 640, 640)
        p3, p4, p5 = backbone_net(input)
        d = head_net(p3, p4, p5)
        print(d)

    def test_new_super_yolo_model(self):
        from tinynetwork.yoloo import SuperYoloo, SuperResolution
        from utils.util import count_parameters, count_trainable_parameters, count_layer_wise_parameters
        my_net = SuperYoloo()
        input = torch.rand(1, 3, 640, 640)
        x, y = my_net(input)

    def test_new_super_yolo_model(self):
        from tinynetwork.yoloo import SuperYoloo, SuperResolution
        from utils.util import count_parameters, count_trainable_parameters, count_layer_wise_parameters

        my_net = SuperYoloo()
        input = torch.rand(1, 3, 640, 640)
        x, y = my_net(input)
        print(f"number of paprameters: {count_parameters(my_net)}")
        print(f"number of trainable parameters: {count_trainable_parameters(my_net)}")
        print(type(x))

    def test_backpropagation_on_my_network(self):
        from tinynetwork.yoloo import SuperYoloo
        import torch
        import torch.nn as nn
        import torch.optim as optim

        my_net = SuperYoloo()

        my_net.train()
        criterion = nn.L1Loss()
        criterion2 = nn.MSELoss()
        optimizer = optim.SGD(my_net.parameters(), lr=0.01)
        input = torch.rand(1, 3, 640, 640)
        target_super = torch.rand(1, 3, 1280, 1280)
        target_yolo1 = torch.rand(1, 144, 80, 80)
        target_yolo2 = torch.rand(1, 144, 40, 40)
        target_yolo3 = torch.rand(1, 144, 20, 20)


        output, output_super = my_net(input)
        loss1 = criterion2(output[0], target_yolo1)
        loss2 = criterion2(output[1], target_yolo2)
        loss3 = criterion2(output[2], target_yolo3)
        loss4 = criterion(output_super, target_super)
        total_loss = 4 * loss1 + 2 * loss2 + loss3 + 1.5 * loss4
        optimizer.zero_grad()
        total_loss.backward()

        for name, param in my_net.named_parameters():
            if param.grad is None:
                print(f"[❌] No gradient for {name}")
                return False
            if torch.isnan(param.grad).any():
                print(f"[❌] NaN in gradient of {name}")
                return False
            if torch.all(param.grad == 0):
                print(f"[❌] All-zero gradient for {name}")
                return False
            print(f"[✅] Gradient OK for {name}")
        return True

    def test_infrence(self):
        from tinynetwork.yoloo import SuperYoloo
        import torch
        import torch.nn as nn
        import torch.optim as optim

        my_net = SuperYoloo(tr_model=False)
        input = torch.rand(1, 3, 640, 640)
        my_net.eval()
        with torch.no_grad():
            output = my_net(input)
            print(output)

    def test_save_model(self):
        from tinynetwork.yoloo import SuperYoloo
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset

        class MyDataSet(Dataset):
            def __init__(self):
                self.data = torch.rand(50, 3, 640, 640)
                self.lable1 = torch.rand(50, 144, 80, 80)
                self.lable2 = torch.rand(50, 144, 40, 40)
                self.lable3 = torch.rand(50, 144, 20, 20)
                self.label4 = torch.rand(50, 3, 1280, 1280)
            def __len__(self):
                return self.data.shape[0]

            def __getitem__(self, item):
                return self.data[item], self.lable1[item], self.lable2[item], self.lable3[item], self.label4[item]


        my_dataset = MyDataSet()
        train_set = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
        my_net = SuperYoloo(tr_model=True).to('cuda')

        criterion = nn.L1Loss()
        criterion2 = nn.MSELoss()

        optimizer = optim.SGD(my_net.parameters(), lr=0.01)
        for epoch in enumerate(range(5)):
            my_net.train()
            train_loss = 0
            for data, lb1, lb2, lb3, lb4 in train_set:
                r_data = data.to('cuda')
                lb1 = lb1.to('cuda')
                lb2 = lb2.to('cuda')
                lb3 = lb3.to('cuda')
                lb4 = lb4.to('cuda')
                output0, output1 = my_net(r_data)
                loss1 = criterion2(output0[0], lb1)
                loss2 = criterion2(output0[1], lb2)
                loss3 = criterion2(output0[2], lb3)
                loss4 = criterion(output1, lb4)
                total_loss = loss1 + loss2 + loss3 + loss4
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': my_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
                # Add any other relevant info
                }

            torch.save(checkpoint, 'checkpoint.pth')

        # checkpoint = torch.load('checkpoint.pth')
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        #
        # model.train()  # Set back to training mode
        # # Continue training from saved epoch


    def test_yolo12_yaml(self):
        from ultralytics import YOLO

        model = YOLO('yolo12n.yaml')  # Loads the YOLOv12-small model architecture
if __name__ == '__main__':
    unittest.main()
