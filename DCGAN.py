import tensorflow as tf
from keras.layers import Conv2D, Input, Dense, Flatten,Dropout, BatchNormalization, UpSampling2D, Conv2DTranspose,Dropout,Reshape,Cropping2D, Activation
from keras.models import Sequential, Model
from keras.layers.advanced_activations import ReLU, LeakyReLU
from keras.activations import sigmoid ,tanh
from keras.optimizers import Adam
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class DCGAN():
    
    def __init__(self, save_path,  kernel_size=(3,3),stride=(2,2),input_size = 200,img_size = (150,150,3),dropout=0.4,
                    optimizer = Adam(beta_1=0.5,lr=0.0002),depth = 3, increment = 32, batch_size = 32):
        """
        save_path (str): path to save generated images
        kernel_size (tuple): size of the kernel for Convolutional layer
        stride (tuple): stride size
        input_size (int): input size of the random noise for generator
        img_size (tuple): size of the images to process
        dropout (float): dropout rate
        depth (int): the depth of the generative and discriminative models, i.e. how many Conv2D layers
        increment (int): how much kernels are added with each depth step
        batch_size (int): amount of images to process each training step

        """
        self._kernel_size = kernel_size
        self._stride = stride
        self._input_size = input_size
        self._img_size = img_size
        self._dropout = dropout
        self._depth = depth
        self._increment = increment
        self._batch_size = batch_size
        self._discr_loss = []
        self._gen_loss = []
        self.save_path = save_path
        self.optimizer = optimizer

    def calc_Incr_Shape(self,dep):

        size = self._img_size[0]
        ks = self._kernel_size[0]
        st = self._stride[0]

        for i in range(dep):
            size = int((size-ks)/st+1)
        return size

    def Generative(self):
        
        units = self.calc_Incr_Shape(self._depth)+2
        model = Sequential()
        
        model.add(Dense(units*units*self._depth*self._increment,input_dim = self._input_size))
        model.add(BatchNormalization(momentum = 0.9))
        model.add(ReLU())
        model.add(Reshape((units,units,self._depth*self._increment)))
        model.add(Dropout(self._dropout))

        for i in range(self._depth-1):
            n_filters = self._increment*(self._depth-i)
            model.add(Conv2DTranspose(n_filters,self._kernel_size, strides = self._stride,padding ='same'))
            model.add(BatchNormalization(momentum = 0.9))
            model.add(ReLU())
        
        model.add(Conv2DTranspose(3,self._kernel_size, strides = self._stride,padding ='same'))
        crop_size = int((model.layers[-1].output_shape[1] - self._img_size[0])/2)
        model.add(Cropping2D(cropping=((crop_size, crop_size), (crop_size, crop_size))))
        #model.add(BatchNormalization(momentum = 0.9))
        model.add(Activation('tanh'))

        model.summary()
        noise = Input(shape=(self._input_size,))
        img = model(noise)

        return Model(noise, img)


    def Discriminative(self):

        model = Sequential()
        for i in range(self._depth):
            n_filters = self._increment*(i+1)
            model.add(Conv2D(n_filters,self._kernel_size, strides =self._stride,padding = 'same'))
            model.add(BatchNormalization(momentum = 0.9))
            model.add(Dropout(self._dropout))
            model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dense(1,activation ='sigmoid'))
        
        img = Input(shape=self._img_size)
        validity = model(img)
        model.summary()

        return Model(img, validity)

    def initialize_GAN(self):
        self.discriminative_model = self.Discriminative()
        self.discriminative_model.compile(loss='binary_crossentropy',optimizer = Adam(beta_1=0.5,lr=0.00005), metrics = ['accuracy'])
        self.discriminative_model.trainable = False
        
        self.generative_model = self.Generative()
        inpt = Input(shape = (self._input_size,))
        gen_img = self.generative_model(inpt)

        is_valid = self.discriminative_model(gen_img)
        self.model = Model(inpt,is_valid)
        self.model.compile(loss='binary_crossentropy', optimizer= self.optimizer,metrics = ['accuracy'])
        return

    def save_imgs(self, epoch,it):
        row, col = 5, 5
        noise = np.random.normal(0, 1, (row * col, self._input_size))
        gen_imgs = self.generative_model.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(row, col)
        count = 0
        for i in range(row):
            for j in range(col):
                axs[i,j].imshow(gen_imgs[count])
                axs[i,j].axis('off')
                count += 1
        fig.savefig(os.path.join(self.save_path,'image_{}_{}.png'.format(epoch,it)))
        plt.close()

    def img_show(self,):
        noise = np.random.normal(0, 1, (3, self._input_size))
        gen_imgs = self.generative_model.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        plt.imshow(gen_imgs[0])
        plt.axis('off')
        plt.show()
        
    def train_Model(self,data,num_epoch,save_interval):

        iterations = data.imgs.shape[0]//self._batch_size
        valid = np.ones((self._batch_size,1))
        fake = np.zeros((self._batch_size,1))

        for epoch in tqdm(range(num_epoch)):
            for it in tqdm(range(iterations)):
                
                img_batch = data.img_Batch(batch_size = self._batch_size)
            
                noise = np.random.uniform(-1.0, 1.0, size=[self._batch_size, self._input_size])

                gen_img = self.generative_model.predict(noise)

                # Train Discriminative model
                disc_loss_real = self.discriminative_model.train_on_batch(img_batch,valid)
                disc_loss_fake = self.discriminative_model.train_on_batch(gen_img,fake)
                disc_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)

                # Train Generative model
                if it%1 ==0:
                    gen_loss = self.model.train_on_batch(noise,valid)

                if it%save_interval ==0:
                    print('Iteration {}\n Discriminative loss: {}\n Discriminative accuracy: {}\n Generative loss: {}'.format(it,disc_loss[0],disc_loss[1],gen_loss[0]))
                    self.save_imgs(epoch,it)
            self._discr_loss.append(disc_loss)
            self._gen_loss.append(gen_loss)







        

