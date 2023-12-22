"""
Author: TLG
"""

import mlflow.pytorch
from tqdm import tqdm

from util.utility import *


def score(params, loader_val, loader_clr, model, uuid, device):
    """
    :param params:
    :param loader_val:
    :param loader_clr:
    :param model:
    :param uuid:
    :param device:
    """
    p_normalized = False
    p_ff = False

    model.eval()

    key_words = ["id", "ood"]
    stats = {}

    # no ensemble
    l2_of_val = {}
    f_val = {}

    for w in key_words:
        features = get_features(loader_val[w], model, normalized=p_normalized, ff=p_ff, device=device)
        f_val[w] = features.numpy()
        l2_of_val[w] = torch.sum(torch.square(features), dim=-1)

    for w in key_words:
        stats[w] = AdvancedStatUpdate()
        stats[w](f_val[w])
        for _ in tqdm(range(params["num_aug"])):
            aug_features = get_features(loader_clr[w], model, normalized=p_normalized, device=device, ff=p_ff)
            stats[w](aug_features.numpy())

    l2_of_mean = {w: np.sqrt(np.sum(np.square(stats[w].mean), axis=-1)) for w in key_words}  # ||Ex||_2
    mean_of_l2 = {w: stats[w].l2_mean for w in key_words}  # E(||x||^2)

    with mlflow.start_run(uuid) as run:
        r = auroc(pos=l2_of_mean["id"], neg=l2_of_mean["ood"])
        mlflow.log_metric("auroc_l2_of_EX", r)

        r = auroc(pos=mean_of_l2["id"], neg=mean_of_l2["ood"])
        mlflow.log_metric("auroc_of_E_square_of_X_2", r)

        r = auroc(pos=l2_of_val["id"], neg=l2_of_val["ood"])
        mlflow.log_metric("auroc_l2_of_val", r)
