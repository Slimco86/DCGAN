# DCGAN
General DCGAN architechture wraped in convinient API

### Simple use case

Simply provide the path containing the images you want to train on, in the main.py specifying the image size (150,150,3) is default.

### Hyperparameters 

The DCGAN class object can take the following parameters during the intializiation :  

* save_path (str): path to save generated images
* kernel_size (tuple): size of the kernel for Convolutional layer
* stride (tuple): stride size
* input_size (int): input size of the random noise for generator
* img_size (tuple): size of the images to process
* dropout (float): dropout rate
* depth (int): the depth of the generative and discriminative models, i.e. how many Conv2D layers
* increment (int): how much kernels are added with each depth step
* batch_size (int): amount of images to process each training step
*  optimizer (func): the optimizer used for generative model

Thus, the model can typically be adapted, based on the needs but the discriminator/generator are symmetric, i.e. the upsampling in generative layer and downsampling in the discriminative layer is the same. 

