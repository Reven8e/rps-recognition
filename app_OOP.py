# © RPS Machine learning- Made by Yuval Simon. For www.bogan.cool

import os, random, cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

class Train_Model():
    def __init__(self):
        self.DIR = "dataset"
        self.CATEGORIES = ['Scissors' ,'Rock', 'Paper']
        self.IMG_SIZE = 64
        self.data = []
        self.x = []
        self.y = []


    def create_dataset(self):
        for cat in self.CATEGORIES:
            path = os.path.join(self.DIR, cat)
            if cat == 'Scissors':
                class_num = 0

            elif cat == 'Rock':
                class_num = 1
            
            elif cat == 'Paper':
                class_num = 2

            for i in tqdm(os.listdir(path)):
                img_array = cv2.imread(os.path.join(path, i))
                img_og = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img_og, (self.IMG_SIZE, self.IMG_SIZE))

                self.data .append([img, class_num])
            
        random.shuffle(self.data)
        return f'\nData Len: {len(self.data)}'


    def create_xy(self):
        for features,label in self.data:
            self.x.append(features)
            self.y.append(label)

        self.x = np.array(self.x)
        self.x = self.x.reshape(-3, 64, 64, 3)
        self.y = np.array(self.y)

        self.x = self.x/255.0

        return f"X Shape: {self.x.shape[1:]}"


    def create_model(self):
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=self.x.shape[1:]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten()) 

        model.add(Dense(64))

        model.add(Dense(3, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        model.fit(self.x, self.y, batch_size=32, epochs=3, validation_split=0.3)

        model.save("rps.h5")



class Test_Model():
    def __init__(self):
        from keras.models import load_model
        self.model = load_model("rps.h5")
        self.IMG_SIZE = 64
        self.CATEGORIES = ['Scissors' ,'Rock', 'Paper']


    def make(self, img):
        global img1
        try:
            img_array = cv2.imread(img)
            img_og = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img1 = cv2.resize(img_og, (self.IMG_SIZE, self.IMG_SIZE))
            return img1.reshape(-3, 64, 64, 3)
        except cv2.error:
            print('Image Preparing Error. (Image/s not found)')
            os._exit(0)


    def test(self):
        try:
            path = input('Please specify dir path to the image/s: ')
            for i in os.listdir(path):
                pred = self.model.predict([self.make(f'{path}/{i}')])
                obj = self.CATEGORIES[np.argmax(pred)]

                plt.imshow(img1, cmap=plt.cm.binary)
                plt.title(obj, fontsize= 18)
                plt.show()
        except FileNotFoundError:
            print('no directory or image/s found.')
            os._exit(0)
        

print('\n\n')
print("1. Train Model.\n2. Test Model.")
option = input('\n:')

if option == '1':
    start = Train_Model()
    start.create_dataset()
    start.create_xy()
    start.create_model()
    os._exit(0)

elif option == '2':
    start = Test_Model()
    start.test()
