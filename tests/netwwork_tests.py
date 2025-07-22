import unittest


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

        l2_yolo_back_bone = torch.rand(1, 64, 160, 160)
        l4_yolo_back_bone = torch.rand(1, 128, 80, 80)
        l7_yolo_back_bone = torch.rand(1, 256, 20, 20)

        superNet = SuperResolution(num_channel=3, c1=64, c2=128, scale_factor=2)
        output_res = superNet(l4_yolo_back_bone, l2_yolo_back_bone)
        print(output_res.shape)

    def test_load_and_getinfo_yolov12n(self):
        from ultralytics import YOLO
        from utils.util import count_parameters, count_trainable_parameters, count_layer_wise_parameters
        import torch

        model = YOLO("/home/my_proj/weights/pretrained_models/yolo12n.pt")
        # x = torch.rand(1, 3, 640, 640)
        # out = model(x)
        # print(out)
        print(model)
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

        model = SuperYoloo(yolo_model_src="/home/my_proj/weights/pretrained_models/yolo12n.pt")
        input = torch.rand(1, 3, 640, 640)
        output, _ = model(input)

    def test_get_output_from_yolo_head(self):
        from ultralytics import YOLO
        import torch
        import torch.nn as nn
        from tinynetwork.yoloo import Concat
        yolo_model = YOLO("/home/my_proj/weights/pretrained_models/yolo12n.pt")

        yolo_model = yolo_model.model.model
        # neck = full_model[9:21]
        neck = nn.Sequential(yolo_model[9], Concat(), yolo_model[11], yolo_model[12], yolo_model[13], yolo_model[14],
                                  yolo_model[15], yolo_model[15], yolo_model[16], yolo_model[17], yolo_model[18], yolo_model[19],
                                  yolo_model[20])
        dummy_input = torch.rand(1, 256, 20, 20)

        x = dummy_input

        for i, layer in enumerate(neck):
            try:
                x = layer(x)
                print(f"Layer {i} output shape: {x.shape}")
            except Exception as error:
                print(f"error: {error}")


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

if __name__ == '__main__':
    unittest.main()
