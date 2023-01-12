import shap
import numpy as np
import torch
from loguru import logger
from torchvision import datasets
from torchvision.transforms import ToTensor
import shap
import numpy as np
from torch.utils.data import DataLoader


logger.add("/tmp/explainer.log")
logger.add("explainer.log")

if __name__ == "__main__":
    from src.settings import ImageExplainerSettings
    from src.visualization import visualize


    logger.info("starting shap_values.py")

    presets = ImageExplainerSettings()

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

    test_dataloader = DataLoader(dataset_test, batch_size=128, shuffle=True)

    model = torch.load(presets.modelname)

    batch = next(iter(test_dataloader))
    images, y = batch

    background = images[:120]
    test_images = images[120:128]
    test_y = y[120:128]

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_images)
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

    class_names = dataset_train.classes

    images_dict = dict()
    for i, l in enumerate(dataset_train.targets):
        if len(images_dict)==10:
            break
        if int(l) not in images_dict.keys():
            images_dict[int(l)] = dataset_train.data[i].reshape((28, 28))

    logger.info(f"Actual labels of tested images: {test_y}")

    visualize.plot_categories(images_dict, class_names)

    shap.image_plot(shap_numpy, -test_numpy)