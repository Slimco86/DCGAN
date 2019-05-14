import os
import cv2
import numpy as np
from sklearn.utils import shuffle

class data_Loader():
    
    def __init__(self, path,img_size):
        self.imgs = []
        self.path = path
        self.labels = []
        self._img_size = img_size
        self.batch=0

        return 

    def load(self):
        "parse the root path with the labled child folders folder"
        folders = os.listdir(self.path)
        print('Reading data')
        for folder in folders:
            if os.path.isdir(folder):
                os.chdir(os.path.join(self.path,folder))
                files = os.listdir()
                for file in files:
                    img = cv2.imread(file)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if img.shape != self._img_size:
                        continue
                    else:
                        self.imgs.append(img.reshape(self._img_size))
                        self.labels.append(folder)
                os.chdir('../')
            else:
                os.chdir(self.path)
                img = cv2.imread(folder)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img.shape != self._img_size:
                    continue
                else:
                    self.imgs.append(img.reshape(self._img_size))

        print('Total of {} images was loaded'.format(len(self.imgs)))
        self.imgs = np.array(self.imgs).reshape(-1,*self._img_size)
        return 

    def shuffle(self,state=101,new=True):
        self.imgs = shuffle(self.imgs,random_state=state)
        self.labels = shuffle(self.labels,random_state=state)
        if new:
            self.batch = 0
        return 'Data was shuffled'

    def img_Batch(self,batch_size):
        self.shuffle(state=np.random.randint(500))
        return self.imgs[0:batch_size]
        
