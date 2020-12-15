# © RPS Machine learning- Made by Yuval Simon. For www.bogan.cool

import os, random, cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

DIR = "dataset"
CATEGORIES = ['scissors' ,'rock']
IMG_SIZE = 64


data = []

def create_dataset():
    for cat in CATEGORIES:
        path = os.path.join(DIR, cat)
        if cat == 'scissors':
            class_num = 0

        elif cat == 'rock':
            class_num = 1
        
        # elif cat == 'paper':
        #     class_num = 2

        for i in tqdm(os.listdir(path)):
            img_array = cv2.imread(os.path.join(path, i))
            img_og = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img_og, (IMG_SIZE, IMG_SIZE))

            data .append([img, class_num])

create_dataset()
random.shuffle(data)
print(f'\nData Len: {len(data)}')

x = []
y = []

for features,label in data:
    x.append(features)
    y.append(label)

x = np.array(x)
x = x.reshape(-3, 64, 64, 3)
y = np.array(y)


x = x/255.0
print(f"X Shape: {x.shape[1:]}")

print('\n\n')
print("1. Train Model.\n2. Test Model.")
option = input('\n:')

if option == '1':
    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=x.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten()) 

    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.fit(x, y, batch_size=32, epochs=3, validation_split=0.3)

    model.save("rps.h5")


elif option == '2':
    from keras.models import load_model

    model = load_model("rps.h5")

    def make(img):
        global IMG_SIZE, img1
        try:
            img_array = cv2.imread(img)
            img_og = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img1 = cv2.resize(img_og, (IMG_SIZE, IMG_SIZE))
            return img1.reshape(-3, 64, 64, 3)
        except cv2.error:
            print('Image Preparing Error. (Image/s not found)')
            os._exit(0)

try:
    path = input('Please specify dir path to the image/s: ')
    for i in os.listdir(path):
        pred = model.predict([make(f'{path}/{i}')])
        if 0 in pred:
            obj = 'Scissors'
        elif 1 in pred:
            obj = 'Rock'
        else:
            obj = 'IDK'
        plt.imshow(img1, cmap=plt.cm.binary)
        plt.title(obj, fontsize= 18)
        plt.show()
except FileNotFoundError:
    print('no directory or image/s found.')
    os._exit(0)
            