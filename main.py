from DCGAN import *
from data_loader import *

Data = data_Loader('D:\\intel-image-classification\\seg_train\\forest',(150,150,3))
Data.load()
"""
GAN = DCGAN('E:\\mschine_learning\\Gan\\gen_images\\',depth = 4, batch_size = 64)
GAN.initialize_GAN()
GAN.train_Model(Data,500,35)
"""
from keras.applications import VGG16


base_model = VGG16(include_top=False,input_shape =(150,150,3))
model = Model(inputs=base_model.input, outputs = base_model.output)
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

prediction = model.predict(Data.imgs[0:5])
print(prediction[0].shape)

