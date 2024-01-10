import shap
import torch
from loguru import logger
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
import sys


logger.add("/tmp/explainer.log")
logger.add("explainer.log")

if __name__ == "__main__":
    from src.settings import ImageExplainerSettings
    from src.visualization import visualize

    logger.info("starting shap_values.py")

    presets = ImageExplainerSettings()

    if sys.argv[1] == "mnist":
        dataset_train = datasets.MNIST(
            root=presets.data_dir,
            train=True,
            download=True,
            transform=ToTensor(),
        )

        dataset_test = datasets.MNIST(
            root=presets.data_dir,
            train=False,
            download=True,
            transform=ToTensor(),
        )
    else:
        dataset_train = datasets.FashionMNIST(
            root=presets.data_dir,
            train=True,
            download=True,
            transform=ToTensor(),
        )

        dataset_test = datasets.FashionMNIST(
            root=presets.data_dir,
            train=False,
            download=True,
            transform=ToTensor(),
        )

    test_dataloader = DataLoader(dataset_test, batch_size=128, shuffle=True)

    model = torch.load(presets.modelname)

    batch = next(iter(test_dataloader))
    images, y = batch

    background = images[:120]
    test_images = images[120:128]
    test_y = y[120:128]
    logger.info(f"Actual labels of tested images: {test_y}")

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_images)
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

    class_names = dataset_train.classes
    images_dict = dict()
    for i, l in enumerate(dataset_train.targets):
        if len(images_dict) == 10:
            break
        if int(l) not in images_dict.keys():
            images_dict[int(l)] = dataset_train.data[i].reshape((28, 28))

    if not presets.imgpath.exists():
        presets.imgpath.mkdir(parents=True)

    imgpath = presets.imgpath / Path("categories.jpeg")
    imgpathshap = presets.imgpath / Path("shap.jpeg")

    visualize.plot_categories(
        images_dict, class_names, figsize=(16, 2), filepath=imgpath
    )

    shap_image = shap.image_plot(shap_numpy, -test_numpy, show=False)
    plt.savefig(imgpathshap)

    def reshape_width(image_to_reshape, image_width):
        factor = image_width.size[0] / image_to_reshape.size[0]
        resized = image_to_reshape.resize(
            (
                int(image_to_reshape.size[0] * factor),
                int(image_to_reshape.size[1] * factor),
            )
        )
        resized.size
        return resized

    imageHeader = Image.open(imgpath)
    imageShap = Image.open(imgpathshap)

    resized_header = reshape_width(imageHeader, imageShap)

    header = asarray(resized_header)
    shap_image = asarray(imageShap)

    logger.info(f"Shape header: {header.shape}")
    logger.info(f"Shape shap image: {shap_image.shape}")

    final_image = np.concatenate((header, shap_image), axis=0)
    imgpathfinal = presets.imgpath / Path("final.jpeg")
    im = Image.fromarray(final_image)
    im.save(imgpathfinal)

    logger.info(f"Saved images to {imgpathfinal}")
