import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        from utils.util import setup_logger
        logger = setup_logger(log_file="/home/my_proj/logs/app1.log")
        logger.debug("This is a debug message.")
        logger.info("This is an info message.")
        logger.warning("This is a warning.")
        logger.error("This is an error.")
        logger.critical("This is critical!")

    def test_reshape_output(self):
        from utils.util import reshape_output
        import torch
        all_tensor = [torch.rand(1, 144, 80, 80), torch.rand(1, 144, 40, 40), torch.rand(1, 144, 20, 20)]

        for item in all_tensor:
            box_pred, obj_pred, cls_pred = reshape_output(item)
            print(box_pred.shape)
            print(obj_pred.shape)
            print(cls_pred.shape)
            print("***********************")

    def test_build_targets(self):
        from utils.util import build_targets
        import torch
        import random

        # targets = torch.rand(3, 5)
        targets = [[random.random() for _ in range(5)] for _ in range(2)]
        shapes = [(80, 80), (40, 40), (20, 20)]
        for sh in shapes:
            obj_target, cls_target, box_target = build_targets(targets=targets, H=sh[0], W=sh[-1])
            print(obj_target.shape)
            print(cls_target.shape)
            print(box_target.shape)
            print("**************************")


if __name__ == '__main__':
    unittest.main()
