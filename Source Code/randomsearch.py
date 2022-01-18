import argparse
import itertools
import json
import math
from datetime import datetime
from pathlib import Path
from random import random

from torch.optim import Adam

from focal import FocalLoss
from loss import *
from main import run_training, basic_validation, basic_training
from skynet import skynet_training, skynet_validation
from models import *


def get_loss_param(loss_str: str):
    alpha = random()
    beta = 1 - alpha
    gamma = 4 * random()
    alpha_combine = random()

    weight = torch.tensor([.7, .1, .05, .15])  # class weight : 70% for background and 15/10/5% for the three others
    loss_obj_dict = {
        "dice_loss": DiceLoss(n_classes=4),
        "jaccard_loss": JaccardLoss(n_classes=4),
        "tversky_loss": TverskyLoss(n_classes=4, alpha=alpha, beta=beta),
        "focal_tversky_loss": FocalTverskyLoss(4, alpha=alpha, beta=beta, gamma=gamma),
        "ce_loss": nn.CrossEntropyLoss(),
        "ce_weighted_loss": nn.CrossEntropyLoss(weight=weight),
        "focal_loss": FocalLoss(alpha=weight, gamma=gamma),
    }

    if " x " in loss_str:
        loss_1, loss_2 = loss_str.split(" x ")
        loss_obj = CombinedLoss(loss_obj_dict[loss_1], loss_obj_dict[loss_2], alpha=alpha_combine)
    else:
        loss_obj = loss_obj_dict[loss_str]

    json_dict = {"name": loss_str}
    if "tversky_loss" in loss_str:
        json_dict["alpha"] = alpha
        json_dict["beta"] = beta
    if "focal" in loss_str:
        json_dict["gamma"] = gamma
    if " x " in loss_str:
        json_dict["alpha_combine"] = alpha_combine

    return json_dict, loss_obj


def get_learning_rate():
    return random() ** 5


def get_batch_size():
    return round((random() + 1) * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobId', type=int)
    parser.add_argument('--jobUid', type=int)
    parser.add_argument('--model', default="SubUNet", type=str)
    parser.add_argument('--loss', default="dice_loss x focal_loss", type=str)
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
        # "batch_size": 8,
        # "batch_size_val": 4,
        # "lr": 0.0001,
        "net": args.model,
        "num_epoch": args.epochs,
        "root_dir": args.root_dir,
        "augment": True,
        "num_classes": 4,
        # "name":,
        # "optimizer":,
        # "loss":,
    }
    print("~~~~~~~~~~~ Creating the CNN model ~~~~~~~~~~")
    train = basic_training
    validate = basic_validation
    if args.model == "UNet":
        model = UNet(n_channels=1, n_classes=args_dict["num_classes"], crop=True)
    elif args.model == "VGG19":
        model = VGG19(n_channels=1, n_classes=args_dict["num_classes"])
    elif args.model == "SegNet":
        model = SegNet(n_channels=1, n_classes=args_dict["num_classes"])
    elif args.model == "PSPNet":
        model = PSPNet(n_channels=1, n_classes=args_dict["num_classes"])
    elif args.model == "SkyNet":
        model = SkyNet(n_channels=1, n_classes=args_dict["num_classes"])
        train = skynet_training
        validate = skynet_validation()
    else:
        args.model = "SubUNet"
        model = SubUNet(n_channels=1, n_classes=args_dict["num_classes"])
        train = skynet_training

    if torch.cuda.is_available():
        model.cuda()

    #### Variable Argument Initialization ####
    loss_json_dict, loss = get_loss_param(args.loss)
    lr = get_learning_rate()
    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    args_dict["lr"] = lr
    args_dict["loss"] = loss
    args_dict["batch_size"] = get_batch_size()
    args_dict["batch_size_val"] = get_batch_size()
    args_dict["name"] = "{m}_{u}-{i}_{l}".format(m=args.model, u=args.jobUid, i=args.jobId, l=loss_json_dict["name"])
    args_dict["optimizer"] = str(optimizer)

    json_dict = args_dict.copy()
    json_dict['loss'], json_dict['optimizer'] = loss_json_dict, 'Adam'

    json_file_name = saves_directory / f'{args_dict["name"]}.json'
    with open(json_file_name, "w") as outfile:
        json.dump(json_dict, outfile)

    folder_name = "Random/{m}_{u}".format(m=args.model, u=args.jobUid)
    start = datetime.now()

    run_training(batch_size=args_dict["batch_size"],
                 batch_size_val=args_dict["batch_size_val"],
                 lr=lr,
                 name=args_dict["name"],
                 folder_name=folder_name,
                 net=model,
                 num_epoch=args_dict["num_epoch"],
                 root_dir=args_dict["root_dir"],
                 augment=args_dict["augment"],
                 optimizer=optimizer,
                 loss=loss,
                 num_classes=args_dict["num_classes"],
                 results_dir=args.output_dir,
                 min_progress_interval=math.inf,  # disables progress bars
                 train=train,
                 validate=validate)

    print(f"Training done in {datetime.now() - start}")
