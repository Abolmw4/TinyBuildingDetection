import unittest


class MyTestCase(unittest.TestCase):
    def test_supernet_input_output(self):
        from tinynetwork.supreres import SuperResolution
        import torch

        superNet = SuperResolution(num_channel=3, c1=128, c2=512, scale_factor=2).to('cuda')
        low_level_feat = torch.rand((1, 128, 64, 64)).to('cuda')
        high_level_feat = torch.rand((1, 512, 16, 16)).to('cuda')

        output_res = superNet(high_level_feat, low_level_feat)
        print(output_res)

    def test_supernet_info(self):
        from tinynetwork.supreres import SuperResolution
        from utils.util import count_parameters, count_trainable_parameters, count_layer_wise_parameters

        superNet = SuperResolution(num_channel=3, c1=128, c2=512, scale_factor=2).to('cuda')

        print(f"All parameter: {count_parameters(superNet)}")

        print(f"trainable parameters: {count_trainable_parameters(superNet)}")

        print(f"layer wise parameters: {count_layer_wise_parameters(superNet)}")
    

if __name__ == '__main__':
    unittest.main()
