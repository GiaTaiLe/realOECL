"""
Author: LGT
"""
import os
import time
import torch
import yaml
import mlflow.pytorch
from termcolor import colored

from util.utility import AverageMeter, ProgressBar


def execute_clr(loader, model, loss_func, optimizer, device, rank, epoch, alpha):
    """
    :param loader:
    :param model:
    :param loss_func:
    :param optimizer:
    :param device:
    :param rank:
    :param epoch:
    :param alpha:
    """
    model.train()

    if "oe" in loader.keys():
        # OECL training
        id_norm = AverageMeter("id_norm", ":.3f")
        oe_norm = AverageMeter("oe_norm", ":.3f")
        loss_meter = AverageMeter("loss", ":.3f")
        progress = ProgressBar(len(loader["id"]), [loss_meter, id_norm, oe_norm], epoch)

        for i, (views_id, views_oe) in enumerate(zip(loader["id"], loader["oe"])):
            # time
            start = time.time()

            # data
            # id
            data_id1, data_id2 = views_id["image"]
            data_id1 = data_id1.to(device)
            data_id2 = data_id2.to(device)

            b, c, h, w = data_id1.size()

            # oe
            data_oe1, data_oe2 = views_oe["image"]
            data_oe1 = data_oe1.to(device)
            data_oe2 = data_oe2.to(device)

            assert data_oe1.size() == data_id1.size()

            # concat
            views = torch.cat([data_id1, data_id2, data_oe1, data_oe2], dim=0)
            views = views.view(-1, c, h, w)

            # forward
            f_fh, f_f = model(views)
            embedding, embedding_oe = f_fh.chunk(2)

            if epoch < 50:
                _alpha = 0.

            else:
                _alpha = alpha

            # loss
            loss, id_n, oe_n = loss_func(embedding=embedding, embedding_oe=embedding_oe, alpha=_alpha)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update meter
            id_norm.update(id_n.item())
            oe_norm.update(oe_n.item())
            loss_meter.update(loss.item())
            if rank == 0:
                progress.display(i, time.time() - start)

        if rank == 0:
            # log
            mlflow.log_metric("train_loss", value=loss_meter.avg, step=epoch)
            mlflow.log_metric("id_norm", value=id_norm.avg, step=epoch)
            mlflow.log_metric("oe_norm", value=oe_norm.avg, step=epoch)

    else:
        # SimCLR training
        loss_meter = AverageMeter("loss", ":.3f")
        progress = ProgressBar(len(loader["id"]), [loss_meter], epoch)

        for i, views in enumerate(loader["id"]):
            # time
            start = time.time()

            # data
            view1, view2 = views["image"]
            view1 = view1.to(device)
            view2 = view2.to(device)
            b, c, h, w = view1.size()

            # re-concat
            views = torch.cat([view1, view2], dim=0)
            views = views.view(-1, c, h, w)

            # forward
            f_fh, latent = model(views)

            # loss
            loss = loss_func(embedding=f_fh)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update meter
            loss_meter.update(loss.item())
            if rank == 0:
                progress.display(i, time.time() - start)

        if rank == 0:
            # log
            mlflow.log_metric("train_loss", value=loss_meter.avg, step=epoch)


def train(params, loader, model, loss_func, optimizer, scheduler, start_epoch, rank, device):
    """
    :param params:
    :param loader:
    :param model:
    :param loss_func:
    :param optimizer:
    :param scheduler:
    :param device:
    :param start_epoch:
    :param rank: to print out info
    :return:
    """
    # ckpt
    print(colored(">>> Load check point PATH for {}".format(params["method"]), "blue"))
    ckpt_key = params["method"] + "_ckpt"
    model_key = params["method"] + "_model"
    yaml_key = params["method"] + "_setup"

    ckpt_path, model_path, yaml_path = params[ckpt_key], params[model_key], params[yaml_key]
    print("Ckpt: {}".format(colored(ckpt_path, "green")))

    if not os.path.exists(yaml_path):
        with open(yaml_path, "w") as fw:
            yaml.dump(params, fw)

    # balancing hyperparameter alpha
    if params["dataset"] in ["imagenet30", "cifar10"]:
        alpha = 1.

    elif params["dataset"] in ["dior", "wbc"]:
        alpha = 0.1

    else:
        raise ValueError("No dataset {}".format(params["dataset"]))

    with mlflow.start_run(run_name="{}".format(params["description"])) as run:
        run_uuid = run.info.run_uuid
        for key_w in params:
            mlflow.log_param(key_w, params[key_w])

        for epoch in range(start_epoch, params["epochs"]):
            init_time = time.time()
            execute_clr(
                loader=loader,
                model=model,
                loss_func=loss_func,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                rank=rank,
                alpha=alpha
            )
            if rank == 0:
                print("One execute: ", time.time() - init_time)
            scheduler.step()
            if rank == 0:
                print("\n")

                # save ckpt
                if epoch % 100 == 0 or epoch > params["epochs"] - 2:
                    torch.save({
                        "data": {"id": params["dataset"]},
                        "optimizer": optimizer.state_dict(),
                        "model": model.state_dict(),
                        "epoch": epoch + 1
                    }, f=ckpt_path)

    return run_uuid
