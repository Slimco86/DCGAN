# DCGAN
General DCGAN architechture implemented in Keras with Tensorflow backend, wraped in convinient API.

The GAN stands for Generative Adversarial Network, and its name is self explanatory. The model consists of the two branches, one generative another is discriminative. The generative branch uses random noise as an input and sequentially transforms this noise into the image. The discriminative branch is used to compare the generated image with the real sample images and to define which one is real and which is fake. The goal of the generative branch, to learn how to "trick" the discriminative branch. The goal of the discriminative is to learn how not to be "tricked". Thus, two branches are competing between each other, establishing a so-called minmax game, which is expressed in minimizing the following loss function.
![picture alt](https://image.slidesharecdn.com/aimeetgans-170110113744/95/generative-adversarial-networks-and-their-applications-9-638.jpg?cb=1484049167)

The discriminative branch is a common, well-known Convolutional network, while the generative branch is its reverse, with Convolutional layers replaced by Convolutional transpose, which results in the upsampling.
![picture alt](https://gluon.mxnet.io/_images/dcgan.png)
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

### Training results

The model was trained on [intel-image-classification database](https://www.kaggle.com/puneet6060/intel-image-classification "intel-image-classification database"), taken from Kaggle. The training is more efficient on single lable data, which is strongly recomended. The seqeunce of images after each 50 epochs is presented below:

