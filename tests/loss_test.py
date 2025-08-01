import unittest


class MyTestCase(unittest.TestCase):
    def test_obj_loss(self):
        from tinyloss.lossfunctions import obj_loss
        import torch
        from utils.util import build_targets, reshape_output
        import random

        targets = [[random.random() for _ in range(5)]]
        one_sample = [torch.rand(1, 144, 80, 80), torch.rand(1, 144, 40, 40), torch.rand(1, 144, 20, 20)]
        loss_obj = 0
        for item in one_sample:
            batch, _, h, w = item.shape
            box_pred, obj_pred, cls_pred = reshape_output(item)
            obj_target, cls_target, box_target = build_targets(targets=targets, H=h, W=w)
            loss_obj += obj_loss(obj_pred=obj_pred, obj_target=obj_pred)
            print(loss_obj)

    def test_cls_loss(self):
        from tinyloss.lossfunctions import cls_loss
        import torch
        from utils.util import build_targets, reshape_output
        import random

        targets = [[random.random() for _ in range(5)]]
        one_sample = [torch.rand(1, 144, 80, 80), torch.rand(1, 144, 40, 40), torch.rand(1, 144, 20, 20)]
        loss_cls = 0
        for item in one_sample:
            batch, _, h, w = item.shape
            box_pred, obj_pred, cls_pred = reshape_output(item)
            obj_target, cls_target, box_target = build_targets(targets=targets, H=h, W=w)
            loss_cls += cls_loss(cls_pred=cls_pred, cls_target=cls_target, obj_target=obj_target)
            print(loss_cls)

    def test_dfl_loss(self):
        from tinyloss.lossfunctions import dfl_loss
        import torch
        from utils.util import build_targets, reshape_output, dfl_loss
        import random

        targets = [[random.random() for _ in range(5)]]
        one_sample = [torch.rand(1, 144, 80, 80), torch.rand(1, 144, 40, 40), torch.rand(1, 144, 20, 20)]
        loss_dfl = 0
        for item in one_sample:
            batch, _, h, w = item.shape
            box_pred, obj_pred, cls_pred = reshape_output(item)
            obj_target, cls_target, box_target = build_targets(targets=targets, H=h, W=w)
            loss_dfl += dfl_loss()
            print(loss_dfl)

if __name__ == '__main__':
    unittest.main()
