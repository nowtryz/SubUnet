import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from medicalDataLoader import MedicalImageDataset
from loss import *
from metrics import *
from models import *
from train_val import *
from utils import *


def run_training(batch_size: int,
                 batch_size_val: int,
                 lr: float,
                 name: str,
                 net: nn.Module,
                 num_epoch: int,
                 root_dir: str,  # path to the dataset
                 augment: bool,
                 optimizer: Optimizer,
                 results_dir="Results",
                 folder_name=None,
                 loss: nn.Module = nn.CrossEntropyLoss(),
                 num_classes=4,
                 save_all_epochs=True,
                 min_progress_interval=.1,
                 train=single_output_training,
                 lr_train=single_lr_training,
                 validate=single_output_validation):

    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    print(' Dataset: {} '.format(root_dir))

    train_set_full = MedicalImageDataset('train_small',
                                         root_dir,
                                         augment=augment,
                                         equalize=False,
                                         load_on_gpu=True,
                                         load_all_dataset=True)

    train_loader_full = DataLoader(train_set_full,
                                   batch_size=batch_size,
                                   worker_init_fn=np.random.seed(),
                                   num_workers=0,
                                   shuffle=True)

    val_set = MedicalImageDataset('val',
                                  root_dir,
                                  equalize=False,
                                  load_on_gpu=True,
                                  load_all_dataset=True)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size_val,
                            worker_init_fn=np.random.seed(),
                            num_workers=0,
                            shuffle=True)

    # Initialize
    #### Create your own model #####
    print(" Model Name: {}, CUDA: {}".format(name, torch.cuda.is_available()))
    print("\tTotal params: {0:,}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))

    #### Metrics ####
    epoch_metrics = Metrics(len(train_loader_full), num_classes=num_classes, loss=loss)
    validation_metrics = Metrics(len(val_loader), num_classes=num_classes, loss=loss)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # lr = torch.tensor(lr, requires_grad=True)
    lr = Variable(torch.tensor(lr), requires_grad=True)

    ### To save statistics ####
    statistics = []
    best_loss_val = 1000
    best_epoch = 0

    if torch.cuda.is_available():
        net.cuda()
        loss.cuda()
        epoch_metrics.cuda()
        validation_metrics.cuda()
        lr.cuda()

    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    results_path = Path(results_dir)
    statistics_directory = results_path/'Statistics' if folder_name is None else results_path/'Statistics'/folder_name
    statistics_directory.mkdir(parents=True, exist_ok=True)
    saves_directory = results_path/'Saves'/name if folder_name is None else results_path/'Saves'/folder_name/name
    saves_directory.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epoch):
        #### Deactivate lr Gradient ####
        lr.requires_grad_(False)
        print(f"lr.requires_grad = {lr.requires_grad}")

        # print(f' ----------  New learning Rate: {lr}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        net.train()
        epoch_tqdm = tqdm(train_loader_full,
                          desc="[Training] Epoch: {} ".format(epoch),
                          mininterval=min_progress_interval)

        # train(net, epoch_tqdm, optimizer, epoch_metrics)
        for batch_idx, data in enumerate(epoch_tqdm):
            images, labels, img_names = data

            ### From numpy to torch variables
            labels = to_var(labels)
            images = to_var(images)

            loss_epoch = train(net, batch_idx, images, labels, epoch_metrics)

            net.zero_grad()
            optimizer.zero_grad()
            loss_epoch.backward()

            # if batch_idx == len(epoch_tqdm) - 1:
            #     print("lr.requires_grad_ set to True")
            #     lr.requires_grad_(True)

            optimizer.step()

        #### Update Learning Rate ####
        net.requires_grad_(False)
        lr.requires_grad_(True)
        print(f"lr.requires_grad = {lr.requires_grad}")
        loss_lr = lr_train(net, images, labels, epoch_metrics)
        loss_lr.backward()
        print(lr.grad)

        # optimizer.add_param_group(lr)
        net.requires_grad_(True)
        lr.requires_grad_(False)
        print(f"lr.requires_grad = {lr.requires_grad}")

        ####  Score metrics  ####
        loss_epoch = epoch_metrics.loss
        iou_epoch = epoch_metrics.iou
        dice_epoch = epoch_metrics.dice
        soft_dice_epoch = epoch_metrics.soft_dice
        epoch_tqdm.set_postfix({
            "loss": loss_epoch,
            "iou": iou_epoch,
            "dice": dice_epoch,
            "soft dice": soft_dice_epoch,
        })

        #### Validation ####
        net.eval()
        val_tqdm = tqdm(val_loader,
                        desc="[Inference] Getting segmentations...".format(epoch),
                        mininterval=min_progress_interval)

        with torch.inference_mode():
            validate(net, val_tqdm, validation_metrics)

        val_tqdm.set_description("[Inference] Segmentation Done !")

        ####  Score metrics  ####
        loss_validation = validation_metrics.loss
        iou_validation = validation_metrics.iou
        dice_validation = validation_metrics.dice
        soft_dice_validation = validation_metrics.soft_dice
        val_tqdm.set_postfix({
            "loss": loss_validation,
            "iou": iou_validation,
            "dice": dice_validation,
            "soft dice": soft_dice_validation,
        })

        statistics.append((epoch,
                           loss_epoch, iou_epoch, dice_epoch, soft_dice_epoch,
                           loss_validation, iou_validation, dice_validation, soft_dice_validation))

        ### Save latest model ####
        if save_all_epochs:
            torch.save(net.state_dict(), saves_directory / f'{epoch}_Epoch.pth')

        if loss_validation < best_loss_val:
            torch.save(net.state_dict(), saves_directory / 'Best_Epoch.pth')
            best_loss_val = loss_validation
            best_epoch = epoch

        print("###  [VAL]  Best Loss : {:.4f} at epoch {}  ###".format(best_loss_val, best_epoch))

        # if epoch % (best_epoch + 100) == 0 and epoch > 0:
        #     for param_group in optimizer.param_groups:
        #         lr *= 0.5
        #         param_group['lr'] = lr
        #         print(' ----------  New learning Rate: {}'.format(lr))

    #### Save statistics ####
    statistics_file = statistics_directory / f"{name}.csv.gz"
    statistics = pd.DataFrame(statistics,
                              columns=["Epoch", "Epoch's loss", "Epoch's  Jaccard index", "Epoch's dice coefficient",
                                       "Epoch's soft dice coefficient", "Validation's loss",
                                       "Validation's  Jaccard index", "Validation's dice coefficient",
                                       "Validation's soft dice coefficient"])
    statistics.set_index("Epoch", inplace=True)
    statistics.to_csv(statistics_file, compression="gzip")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="SubUNet", type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--batch_size_val', default=4, type=int)
    parser.add_argument('--augment', default=True, type=bool)
    parser.add_argument('-d', '--root-dir', default="./Data/", type=str)
    parser.add_argument('-o', '--output-dir', default="./Results/", type=str)
    parser.add_argument('--num-classes', default=4, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    args = parser.parse_args()

    train = single_output_training
    lr_train = single_lr_training
    validate = single_output_validation

    print("~~~~~~~~~~~ Creating the CNN model ~~~~~~~~~~")
    if args.model == "UNet":
        model = UNet(n_channels=1, n_classes=args.num_classes, crop=True)
    elif args.model == "VGG19":
        model = VGG19(n_channels=1, n_classes=args.num_classes)
    elif args.model == "SegNet":
        model = SegNet(n_channels=1, n_classes=args.num_classes)
    elif args.model == "PSPNet":
        model = PSPNet(n_channels=1, n_classes=args.num_classes)
    elif args.model == "SkyNet":
        model = SkyNet(n_channels=1, n_classes=args.num_classes)
        train = multi_output_training
        lr_train = multi_lr_training
    elif args.model == "SubUNet":
        model = SubUNet(n_channels=1, n_classes=args.num_classes)
        train = multi_output_training
        lr_train = multi_lr_training
    else:
        raise NotImplementedError('Unknown model to train:', args.model)

    #### Loss Initialization ####
    weight = torch.tensor([.7, .1, .05, .15])  # class weight : 70% for background and 15/10/5% for the three others
    focal_loss = FocalLoss(alpha=weight, gamma=2)
    tversky_loss = FocalTverskyLoss(n_classes=args.num_classes, alpha=0.1, beta=0.9, gamma=2)
    loss = CombinedLoss(focal_loss, tversky_loss, alpha=.5)

    if torch.cuda.is_available():
        model.cuda()

    #### other stuff ####
    name_with_date = args.model + ' ' + datetime.now().strftime("%Y-%m-%d %H-%M")
    train_optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    warnings.filterwarnings("ignore")
    run_training(batch_size=args.batch_size,
                 batch_size_val=args.batch_size_val,
                 lr=args.lr,
                 net=model,
                 name=name_with_date,
                 num_epoch=args.epochs,
                 root_dir=args.root_dir,  # path to the dataset
                 augment=args.augment,
                 num_classes=args.num_classes,
                 optimizer=train_optimizer,
                 loss=loss,
                 results_dir=args.output_dir,
                 train=train,
                 lr_train=lr_train,
                 validate=validate)
