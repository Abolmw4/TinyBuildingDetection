import wandb

from utils.util import train_tinynet, validation_tinynet, load_json, setup_logger, check_device
from tinydata.tinydataset import TinyData
from tinynetwork.yoloo import SuperYoloo
from torch.utils.data import DataLoader
import os
from tinynetwork.yoloo import SuperYoloo, SuperResolution
from datetime import datetime
import torch
from torch.optim import Adam, SGD

__LOGGER = setup_logger(log_file=os.path.join("/home/my_proj/logs",f"superyolo{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log"))

def main():
    # wandb.login()
    __LOGGER.info("in main function ....")
    config = load_json()
    __LOGGER.debug("config -> json config -> set successful")

    batch, workers, world_size = config.get("batch_size"), config.get("number_of_workers"), config.get("world_size")
    number_of_workers = min([os.cpu_count() // world_size, batch if batch > 1 else 0, workers])
    device = check_device(config.get("device"))
    __LOGGER.debug("hyperparams(bathc, workers, world_size, device) -> set successful")


    tiny_data = TinyData(data_dir=config.get("dataset_dir"), img_size=config.get("image_size"))

    dataset = DataLoader(dataset=tiny_data, batch_size=batch, num_workers=number_of_workers, pin_memory=True,
                         collate_fn=TinyData.collate_fn, drop_last=True)

    __LOGGER.debug("dataset & dataloader -> set successful")

    my_net = SuperYoloo().to(device)
    __LOGGER.debug("model -> SuperYolo set successful")

    optimizer = Adam(params=my_net.parameters(), lr=config.get("learning_rate"))
    __LOGGER.debug("optimizer -> Adam set successful")
    for epoch in range(config.get("number_of_epoch")):
        __LOGGER.debug(f"train -> epoch {epoch} started....")
        my_net.train()
        x, y, w, z, f = train_tinynet(model=my_net, data_loader=dataset, optimizer=optimizer)
        __LOGGER.debug(f"validation -> epoch {epoch + 1} started....")
        validation_tinynet(model=my_net, data_loader=dataset, device="cuda")
        __LOGGER.debug(f"train & validation -> epoch {epoch + 1}  finished.")




if __name__ == "__main__":
    main()
