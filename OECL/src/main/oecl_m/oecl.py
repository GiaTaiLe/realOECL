"""
Author: TLG
"""
import argparse

import torch.distributed as dist
import torch.nn as nn

from engine.oecl_m.config import PathConfig
from engine.oecl_m.test import *
from engine.oecl_m.train import train
from engine.oecl_m.wrapper import *
from util.utility import set_random_seed

# parser
parser = argparse.ArgumentParser(description="SimCLR_OE or OECL")
parser.add_argument("--config_env", help="Config file for the environment")
parser.add_argument("--config_exp", help="Config file for experiment")
parser.add_argument("--times", help="Config file for experiment")
parser.add_argument("--id_class", help="Config file for experiment")
parser.add_argument("--m", default=False, help="Config file for experiment")
parser.add_argument("--seed", help="Config file for experiment")
parser.add_argument("--ddp", default=True, help="Train with Distributed Data Parallel")
# parser.add_argument("--test_mode", default="auroc", help="Test: auroc, plot")

args = parser.parse_args()

local_rank = 0

if args.ddp:
    local_rank = int(os.environ["LOCAL_RANK"])
    # print("Local device {}".format(local_rank))


def main(rank=local_rank):
    """
    :param rank: for ddp setup
    :return:
    """
    # config
    config_path = PathConfig(args.config_env, args.config_exp, args.times, int(args.id_class))
    params = config_path.create_config()

    if params["dataset"] == "cifar10":
        params["backbone"] = "resnet18_cifar10"

    # device
    print(colored(">>Setup device:", "blue"))
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    print(">>Device: ", colored("{}".format(device), "green"), "\n")

    world_size = 1
    print(colored(">>Set up world_size for ddp: {}".format(world_size), "blue"))
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    set_random_seed(int(args.seed))

    # model
    device = torch.device("cuda:{}".format(rank))
    model = get_model(params, params["final_dim"])
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    # wrap model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    print(colored(">>Build model for {}".format(device), "blue"))
    print(">>Model: {}".format(colored(model.__class__.__name__, "green")), "\n")

    # data
    # id
    train_id = get_data(params=params, train=True, category="id", trn="simclr")

    data_id_sampler = torch.utils.data.DistributedSampler(
        train_id,
        num_replicas=world_size,
        rank=rank
    )

    loader_id = torch.utils.data.DataLoader(dataset=train_id, batch_size=params["batch_size"],
                                            num_workers=params["num_workers"],
                                            pin_memory=True, sampler=data_id_sampler, drop_last=True,

                                            persistent_workers=False)

    # oe
    train_oe = get_data(params=params, train=True, category="oe", trn="simclr")

    data_oe_sampler = torch.utils.data.DistributedSampler(
        train_oe,
        num_replicas=world_size,
        rank=rank
    )

    loader_oe = torch.utils.data.DataLoader(dataset=train_oe, batch_size=params["batch_size"],
                                            num_workers=params["num_workers"],
                                            pin_memory=True, sampler=data_oe_sampler, drop_last=True,
                                            persistent_workers=False)

    loader_train = {"id": loader_id, "oe": loader_oe}

    # map location for load ckpt from cuda:0 to current cuda:local_rank
    map_location = {"cuda:0": "cuda:{}".format(local_rank)}

    # Notifications about loaded data.
    print(">> Load successfully {} dataset {} with {} samples".format(
        colored("training ID", "blue"),
        colored(params["dataset"], "green"),
        colored(str(len(train_id)), "green")
    ))

    # loss function
    print(colored(">>Build loss function", "blue"))
    loss_func = get_criterion(params=params)
    print(">>Criterion: {}".format(colored(loss_func.__class__.__name__, "green")), "\n")

    # optimizer
    print(colored(">>Build optimizer:", "blue"))
    optimizer = get_optimizer(params=params, model=model)
    print(">>Optimizer:  {}\n".format(colored(optimizer.__class__.__name__, "green")))

    # scheduler
    print(colored(">>Build scheduler:", "blue"))
    scheduler = get_scheduler(params, optimizer)
    print(">>Scheduler:  {}\n".format(colored(scheduler.__class__.__name__, "green")))

    # ckpt
    ckpt_path = params["{}_ckpt".format(params["method"])]
    if os.path.exists(ckpt_path):
        print(colored(">>Restart from CHECKPOINT: ", "blue"), end="")
        print("ckpt: {}".format(colored(ckpt_path, "green")))
        ckpt = torch.load(ckpt_path, map_location=map_location)
        optimizer.load_state_dict(ckpt["optimizer"])
        model.load_state_dict(ckpt["model"])
        model = model.to(device)
        start_epoch = ckpt["epoch"]

    else:
        print(colored(">>No ckpt file at", "blue"))
        print("ckpt: {}".format(colored(ckpt_path, "green")))
        start_epoch = 0

    # run
    run_uuid = train(
        params=params,
        loader=loader_train,
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        start_epoch=start_epoch,
        rank=rank
    )

    if rank == 0:
        # test and save on device_rank = 0
        id_val = get_data(params=params, train=False, category="id", trn="val", formatter=1)
        loader_id_val = get_dataloader(params, id_val, drop_last=False, shuffle=False)

        ood_val = get_data(params=params, train=False, category="ood", trn="val", formatter=1)
        loader_ood_val = get_dataloader(params, ood_val, drop_last=False, shuffle=False)

        id_train = get_data(params=params, train=True, category="id", trn="val", formatter=1)
        loader_id_train = get_dataloader(params, id_train, drop_last=False, shuffle=False)

        loader_val = {"id": loader_id_val, "ood": loader_ood_val, 'train': loader_id_train}

        # load dara with simclr transform.
        id_clr = get_data(params=params, train=False, category="id", trn="simclr", formatter=1)
        loader_id_clr = get_dataloader(params, id_clr, drop_last=False, shuffle=False)

        ood_clr = get_data(params=params, train=False, category="ood", trn="simclr", formatter=1)
        loader_ood_clr = get_dataloader(params, ood_clr, drop_last=False, shuffle=False)

        loader_clr = {"id": loader_id_clr, "ood": loader_ood_clr}

        # test and save on device_rank = 0
        score(params=params, loader_clr=loader_clr, loader_val=loader_val, model=model, uuid=run_uuid, device=device)


if __name__ == "__main__":
    print(colored("Start the program", "blue"))
    main()
