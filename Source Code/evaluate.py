""""Model performance visual evaluation

This script allows to visually compare predictions from a model and the
ground truth by drawing the segmentation on the image. All the images of
the validation dataset are computed and then stored on hard drive.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models import UNet
from medicalDataLoader import MedicalImageDataset
from utils import getTargetSegmentation
from matplotlib.colors import to_rgb, TABLEAU_COLORS
from scipy.ndimage import sobel
from PIL import Image
from pathlib import Path, PurePath

# left ventricle: 3
left_ventricle_colour = np.asarray(to_rgb(TABLEAU_COLORS['tab:orange']))  # could be blue
# right ventricle: 1
right_ventricle_colour = np.asarray(to_rgb(TABLEAU_COLORS['tab:cyan']))  # could be red
# myocardium: 2
myocardium_colour = np.asarray(to_rgb(TABLEAU_COLORS['tab:green']))  # could be green

red = np.asarray(to_rgb(TABLEAU_COLORS['tab:red']))
pred_mask_enhancement = 1


def edges(mask):
    mask = np.asarray(mask)
    augmented_mask = mask * 255  # convert to black and white mask for better sobel performances
    sx = sobel(augmented_mask, axis=0, mode='constant')
    sy = sobel(augmented_mask, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    sob_max = np.max(sob)
    return (sob > (sob_max / 2.0)) & mask  # not to go beyond the contours of segmentation


def main(weights, root_dir, model_name):
    with torch.inference_mode():
        # Load model abd weights
        model = UNet(n_channels=1, n_classes=4)
        model.load_state_dict(torch.load(weights))
        model.eval()

        # Initialize dataset
        transformer = transforms.ToTensor()
        val_set = MedicalImageDataset('val', root_dir, transformer, transformer)
        val_loader = DataLoader(val_set, shuffle=False)

        # Target directory where images will be saved
        directory = Path.cwd() / 'Results/Images' / model_name / PurePath(weights).name
        directory.mkdir(parents=True, exist_ok=True)

        # Iterate over all images of the validation dataset
        for i, data in enumerate(val_loader):

            images, labels, img_names = data
            image = images[0, 0]  # image 0, channel 0
            label = labels[0, 0]

            # Retrieve Ground truth and predictions from the model
            gt = getTargetSegmentation(label)
            pred = model(images)
            pred = pred.detach().numpy()
            pred = np.argmax(pred[0], axis=0)

            # prepare the base image for manipulations
            res = np.asarray(image)  # From Tensor to ndarray
            res = np.expand_dims(res, 2)  # Creating a third dimension for colours
            res = np.repeat(res, 3, axis=2)  # Fill the 3 colours channels with the grey scale

            # Add segmentation highlights to the image
            res[pred == 1] *= right_ventricle_colour * pred_mask_enhancement
            res[pred == 2] *= myocardium_colour * pred_mask_enhancement
            res[pred == 3] *= left_ventricle_colour * pred_mask_enhancement
            res[edges(pred == 1)] = right_ventricle_colour
            res[edges(pred == 2)] = myocardium_colour
            res[edges(pred == 3)] = left_ventricle_colour
            res[edges(gt == 1)] = red
            res[edges(gt == 2)] = red
            res[edges(gt == 3)] = red

            # Convert from normalized image to 3-bytes pixels image
            res *= 255
            res = res.astype(np.uint8)

            # save the image
            filename = directory / PurePath(img_names[0]).name
            print(f'Saving {filename}...')
            im = Image.fromarray(res.astype(np.uint8), mode='RGB')
            im.save(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--weights-file", type=str)
    parser.add_argument('-d', '--root-dir', default="../Data/", type=str)
    parser.add_argument('--model-name', default="ModelStats", type=str)
    args = parser.parse_args()
    main(weights=args.weights_file,
         root_dir=args.root_dir,
         model_name=args.model_name)
