# Explainable AI for images

In the previous notebook you've seen that we can explain tabular data by explaining which feature(s) have the most impact on the prediction and in which direction they pull the prediction. We can do this for the entire dataset or for one single prediction. 

Run the mnist or mnistfashion (make mnist/make mnistfashion) and you will see that we can also explain why a certain image was categorized as something, in these two datasets either a number for MNIST or clothing category for MNISTFasion. 

The first file (02_image_explainer.py) makes an CNN for either one of these two datasets.

The second file (03_shap_values.py) makes an explainer on top of this model. The file first loads the model. Then some training examples are selected as background and some test images that we can inspect the SHAP values on. Then we calculate shap values which can then be plotted (when running this file, this image will be stored in img/shap.jpeg).

If you would run the MNIST, the 'columns' of the shap.jpeg image can be understood by just interpreting the columns from left to right as 0123456789, but when running the FashionMNIST, this is not that obvious. Therefore it would be great if we can add a header with categorynames and an example of this category. This can be done with the visualize.plot_categories() function. The header will be saved ad categories.jpeg. 

Now we have two pictures, but it would be convenient if we can add them on top of eachother. Two pictures can be combined vertically (on top of each other) by calling np.concatenate with axis = 0, but for this to work, the width of both pictures needs to be equal. The resize_width() function performs this action. The final image is saved in img/final.jpeg. 

Some comments what can be understood from this picture:
- Positive shap values are denoted by red color and they represent the pixels that contributed to classifying that image as that particular class.
- Negative shap values are denoted by blue color and they represent the pixels that contributed to NOT classify that image as that particular class.
- Each row contains each one of the test images we computed the shap values for.
- Each column represents the ordered categories that the model could choose from. Notice that shap.image_plot just makes a copy of the classified image, but we can use the plot_categories function in src/ created earlier to show an example of that class for reference.

Read more about SHAP [here](https://shap.readthedocs.io/en/latest/)

# Running the files
You should be able to run all experiments with `make mnist` or `make mnistfashion`:
- go to your shell
- use cd or z to go into the directory where you can find the Makefile
- We want to be able to use both datasets, therefore there are two different `make` commands: `mnist` and `mnistfashion`. run either one of them from the shell

Make mnist:
- make model: creates the model based with dataset MNIST
- make shap: calculates shap values and plots them in a picture including headers
- make tb: opens tensorboard to check the model performance

Make mnistfashion:
- make model: creates the model based with dataset FashionMNIST
- make shap: calculates shap values and plots them in a picture including headers
- make tb: opens tensorboard to check the model performance

The Makefile consists just of commands you would otherwise run from the commandline. You can read more about them [here](https://opensource.com/article/18/8/what-how-makefile)