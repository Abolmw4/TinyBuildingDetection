from utils.util import train_tinynet
from tinydata.tinydataset import TinyData
from tinynetwork.yoloo import SuperYoloo
from torch.utils.data import DataLoader
import os
from tinynetwork.yoloo import SuperYoloo, SuperResolution


def main():
    tiny_data = TinyData(data_dir="/home/my_proj/dataset/Dr_dataset", img_size=1024)

    batch, workers, world_size = 4, 1, 1

    batch_size = min(batch, len(tiny_data))

    number_of_workers = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])
    dataset = DataLoader(dataset=tiny_data, batch_size=batch_size, num_workers=number_of_workers, pin_memory=True, collate_fn=TinyData.collate_fn)

    my_net = SuperYoloo()
    train_tinynet(model=my_net, data_loader=dataset)


if __name__ == "__main__":
    main()
