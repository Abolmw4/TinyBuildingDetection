import os
import unittest

import numpy as np



class MyTestCase(unittest.TestCase):
    def test_dataset_info(self):
        import os
        count, lbl_count = 0, 0

        for item in os.listdir("/home/my_proj/dataset/Dr_dataset/images"):
            if os.path.isfile(os.path.join("/home/my_proj/dataset/Dr_dataset/images", item)):
                count += 1
        for item in os.listdir("/home/my_proj/dataset/Dr_dataset/labels"):
            if os.path.isfile(os.path.join("/home/my_proj/dataset/Dr_dataset/labels", item)):
                lbl_count += 1
        print(f"number of data: {count}")
        self.assertEqual(count, lbl_count)

    def test_tinydataset(self):
        from tinydata.tinydataset import TinyData
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        batch, workers, world_size = 32, 1, 1

        tiny_data = TinyData(data_dir="/home/my_proj/dataset/Dr_dataset")

        batch_size = min(batch, len(tiny_data))

        number_of_workers = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])
        print(number_of_workers)
        print("****** Dataset info ******")
        print(f"number of data: {len(tiny_data)} | batch_size: {batch_size} | workers: {number_of_workers}")

        dataset = DataLoader(dataset=tiny_data, batch_size=batch_size, num_workers=number_of_workers, pin_memory=True, collate_fn=TinyData.collate_fn)
        for item in tqdm(dataset):
            img, labels_out, img_adress, shapes = item
            print(f"dataset output img:{img.shape} | labels: {labels_out.shape} | img_adress: {len(img_adress)} | shapes: {shapes}")
            break

if __name__ == '__main__':
    unittest.main()
