The CNN is built using CNN. It uses the FashionMNIST dataset which was built on the platform of MNIST dataset. The dataset contains a collection of 70,000 pictures (1, 28, 28) of dress, foot-wears etc. The CNN achieves nearly 97% accuracy in training set and more than 90% accuracy in test set with 30 epochs trained on one GPU.  

The CNN uses three convolution layers and three fully connected layers. Two special classes - RunBuilder &amp; RunManager were built to test various netoworks with different set of hyperparameters. These classes have also been incorporated in the code. Any new parameter can be easily added using the params dictionary and making few changes. The RunManager class stores the values for each run which can be used in tensorboard to visualize the impact of changing hyperparameters. The results are also saved in a .csv and .json file to be utilized later. This class also enables the user to visualize the accuracy with respect to parameters in real time while training. 
