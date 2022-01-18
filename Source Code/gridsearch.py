import argparse
import itertools
import json
import math
from datetime import datetime
from pathlib import Path

from torch.optim import Adam

from focal import FocalLoss
from loss import *
from main import run_training, basic_training, basic_validation
from models import *
from skynet import skynet_training, skynet_validation


def get_loss(job_id: int, alpha=0.1, beta=0.9, gamma=2):
    #### Loss Initialization ####
    weight = torch.tensor([.7, .1, .05, .15])  # class weight : 70% for background and 15/10/5% for the three others
    group_loss_constr = {
        "dice_loss": DiceLoss(n_classes=4),
        "jaccard_loss": JaccardLoss(n_classes=4),
        "tversky_loss": TverskyLoss(n_classes=4, alpha=alpha, beta=beta),
        "focal_tversky_loss": FocalTverskyLoss(4, alpha=alpha, beta=beta, gamma=gamma)
    }
    pts_loss_constr = {
        "ce_loss": nn.CrossEntropyLoss(),
        "ce_weighted_loss": nn.CrossEntropyLoss(weight=weight),
        "focal_loss": FocalLoss(alpha=weight, gamma=gamma)
    }

    group_loss = list(group_loss_constr.keys())
    pts_loss = list(pts_loss_constr.keys())

    nb_g_loss, nb_p_loss = len(group_loss), len(pts_loss)
    nb_total_loss = nb_g_loss + nb_p_loss + nb_g_loss * nb_p_loss

    job_id = job_id % nb_total_loss
    if job_id < nb_g_loss:
        json_dict = {"name": group_loss[job_id]}
        if "tversky_loss" in json_dict["name"]:
            json_dict["alpha"] = alpha
            json_dict["beta"] = beta
        if json_dict["name"] == "focal_tversky_loss":
            json_dict["gamma"] = gamma
        return json_dict, group_loss_constr[json_dict["name"]]

    elif job_id < nb_g_loss + nb_p_loss:
        json_dict = {"name": pts_loss[job_id - nb_g_loss]}
        if json_dict["name"] == "focal_loss":
            json_dict["gamma"] = gamma
        return json_dict, pts_loss_constr[json_dict["name"]]

    else:
        l1, l2 = list(itertools.product(group_loss, pts_loss))[job_id - nb_g_loss - nb_p_loss]
        json_dict = {"name": l1 + " x " + l2}
        if "tversky_loss" in json_dict["name"]:
            json_dict["alpha"] = alpha
            json_dict["beta"] = beta
        if "focal" in json_dict["name"]:
            json_dict["gamma"] = gamma
        return json_dict, CombinedLoss(group_loss_constr[l1], pts_loss_constr[l2], alpha=alpha)


def get_model(job_id: int, n_channels=1, n_classes=4):
    models = {
        # "UNet": lambda: UNet(n_channels=n_channels, n_classes=n_classes, crop=True),
        # "VGG19": lambda: VGG19(n_channels=n_channels, n_classes=n_classes),
        # "SegNet": lambda: SegNet(n_channels=n_channels, n_classes=n_classes),
        # "PSPNet": lambda: PSPNet(n_channels=n_channels, n_classes=n_classes),
        # "SkyNet": lambda: SkyNet(n_channels=n_channels, n_classes=n_classes),
        "SubUNet": lambda: SubUNet(n_channels=n_channels, n_classes=n_classes),
    }
    model_names = list(models.keys())
    job_id = job_id % len(models)
    name = model_names[job_id]

    train = skynet_training if name in ("SkyNet", "SubUNet") else basic_training
    validate = skynet_validation if name == "SkyNet" else basic_validation

    return model_names[job_id], models[model_names[job_id]], train, validate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobId', type=int)
    parser.add_argument('--jobUid', type=int)
    # parser.add_argument('--model', default="UNet", type=str)
    # parser.add_argument('--searchType', default="Grid", type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('-d', '--root-dir', default="./Data/", type=str)
    parser.add_argument('-o', '--output-dir', default="./Results/", type=str)
    args = parser.parse_args()

    # if args.searchType == "Grid" or args.searchType == "Random":
    #     use_grid_search = args.searchType == "Grid"
    # else:
    #     raise RuntimeError("Invalide search type: {} != Grid or Random".format(args.searchType))

    saves_directory = Path(args.output_dir) / 'Search'
    saves_directory.mkdir(parents=True, exist_ok=True)

    #### Fixed Argument Initialization ####
    args_dict = {
        "batch_size": 8,
        "batch_size_val": 4,
        "lr": 0.0001,
        # "net": args.model,
        "num_epoch": args.epochs,
        "root_dir": args.root_dir,
        "augment": True,
        "num_classes": 4,
        # "name":,
        # "optimizer":,
        # "loss":,
    }

    print("~~~~~~~~~~~ Creating the CNN model ~~~~~~~~~~")
    model_name, model, train, validate = get_model(args.jobId, n_channels=1, n_classes=args_dict["num_classes"])
    model = model()

    if torch.cuda.is_available():
        model.cuda()

    #### Variable Argument Initialization ####
    loss_json_dict, args_dict["loss"] = get_loss(args.jobId)
    args_dict["net"] = model_name
    args_dict["name"] = "{m}_{u}-{i}_{l}".format(m=model_name, u=args.jobUid, i=args.jobId, l=loss_json_dict["name"])
    args_dict["optimizer"] = Adam(model.parameters(), lr=args_dict["lr"], betas=(0.9, 0.99))

    json_dict = args_dict.copy()
    json_dict['loss'], json_dict['optimizer'] = loss_json_dict, 'Adam'

    json_file_name = saves_directory / f'{model_name}.json'
    with open(json_file_name, "w") as outfile:
        json.dump(json_dict, outfile)

    folder_name = "job-{u}/{m}".format(m=model_name, u=args.jobUid, i=args.jobId, l=loss_json_dict["name"])
    start = datetime.now()

    run_training(batch_size=args_dict["batch_size"],
                 batch_size_val=args_dict["batch_size_val"],
                 lr=args_dict["lr"],
                 name=args_dict["name"],
                 folder_name=folder_name,
                 net=model,
                 num_epoch=args_dict["num_epoch"],
                 root_dir=args_dict["root_dir"],
                 augment=args_dict["augment"],
                 optimizer=args_dict["optimizer"],
                 loss=args_dict["loss"],
                 num_classes=args_dict["num_classes"],
                 results_dir=args.output_dir,
                 min_progress_interval=math.inf,  # disables progress bars
                 train=train,
                 validate=validate)

    print(f"Training done in {datetime.now() - start}")
