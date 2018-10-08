# This file is a part of StarNet code.
# https://github.com/nekitmm/starnet
# 
# StarNet is a neural network that can remove stars from images leaving only background.
# 
# Throughout the code all input and output images are 8 bits per channel tif images.
# This code in original form will not read any images other than these (like jpeg, etc), but you can change that if you like.
# 
# Copyright (c) 2018 Nikita Misiura
# http://www.astrobin.com/users/nekitmm/
# 
# This code is distributed on an "AS IS" BASIS WITHOUT WARRANTIES OF ANY KIND, express or implied.
# Please review LICENSE file before use.

import numpy as np
from PIL import Image as img
from scipy.misc import toimage
import matplotlib.pyplot as plt
import sys
from os import listdir
from os.path import isfile, join

WINDOW_SIZE = 256                      # Size of the image fed to net. Do not change until you know what you are doing! Default is 256
                                       # and changing this will force you to train the net anew.

def list_train_images(addr):
    # this function will open only .tif images (8 bit per channel)
    original_files = [f for f in listdir(addr + "/original/") if isfile(join(addr + "/original/", f)) and f.endswith(".tif")]
    starless_files = [f for f in listdir(addr + "/original/") if isfile(join(addr + "/original/", f)) and f.endswith(".tif")]
    assert(len(original_files) == len(starless_files))
    for i in range(len(original_files)):
        assert(original_files[i] == starless_files[i])
    return original_files
    
def open_train_images(addr, list, max):
    original = []
    starless = []
    if max >= len(list):
        for i in list:
            original.append(np.array(img.open(addr + "/original/" + i), dtype = np.float32))
            starless.append(np.array(img.open(addr + "/starless/" + i), dtype = np.float32))
        return original, starless
    else:
        used = []
        for i in range(max):
            ind = int(np.random.rand() * len(list))
            if ind in used:
                while ind in used:
                    ind = int(np.random.rand() * len(list))
            used.append(ind)
            original.append(np.array(img.open(addr + "/original/" + list[ind]), dtype = np.float32))
            starless.append(np.array(img.open(addr + "/starless/" + list[ind]), dtype = np.float32))
        return original, starless
    
def list_test_images(addr):
    # this function will open only .tif images (8 bit per channel)
    files = [f for f in listdir(addr) if isfile(join(addr, f)) and f.endswith(".tif")]
    return files

def get_input_img(Xinp, Yinp, size = WINDOW_SIZE, rotate = 0, resize = 1):
    assert(Xinp.shape == Yinp.shape)
    if rotate != 0:
        Xim = img.fromarray(np.uint8(Xinp))
        Yim = img.fromarray(np.uint8(Yinp))
        Xim = Xim.rotate(rotate, resample = img.BICUBIC)
        Yim = Yim.rotate(rotate, resample = img.BICUBIC)
        Xinp = np.array(Xim)
        Yinp = np.array(Yim)
    h, w, _ = Xinp.shape
    if resize != 1 and h > 600 and w > 600:
        h = np.int(h * resize)
        w = np.int(w * resize)
        Xim = img.fromarray(np.uint8(Xinp))
        Yim = img.fromarray(np.uint8(Yinp))
        Xim = Xim.resize((w, h), resample = img.BICUBIC)
        Yim = Yim.resize((w, h), resample = img.BICUBIC)
        #Xim.save('./x.png')
        #Yim.save('./y.png')
        Xinp = np.array(Xim)
        Yinp = np.array(Yim)
    y = int(np.random.rand() * (h - size))
    x = int(np.random.rand() * (w - size))
    return (np.array(Xinp[y:y + size, x:x + size, :]) / 255.0 - 0.0, np.array(Yinp[y:y + size, x:x + size, :]) / 255.0 - 0.0)

def get_input_img_with_augmentation(Xinp, Yinp, size = WINDOW_SIZE):
    # rotate with arbitrary angle
    if np.random.rand() < 0.33:
        r = np.random.randint(360)
    else:
        r = 0
    if np.random.rand() < 0.33:
        s = 0.5 + np.random.rand() * 1.5
    else:
        s = 1
    (X_, Y_) = get_input_img(Xinp, Yinp, size, rotate = r, resize = s)

    # flip horizontally
    if np.random.rand() < 0.5:
        X_ = np.flip(X_, axis = 1)
        Y_ = np.flip(Y_, axis = 1)
    # flip vertically
    if np.random.rand() < 0.5:
        X_ = np.flip(X_, axis = 0)
        Y_ = np.flip(Y_, axis = 0)
    # rotate 90, 180 or 270
    if np.random.rand() < 0.5:
        k = int(np.random.rand() * 3 + 1)
        X_ = np.rot90(X_, k, axes = (1, 0))
        Y_ = np.rot90(Y_, k, axes = (1, 0))
    # turn into BW
    if np.random.rand() < 0.1:
        Xm = np.mean(X_, axis = 2, keepdims = True)
        Ym = np.mean(Y_, axis = 2, keepdims = True)
        X_ = np.concatenate((Xm, Xm, Xm), axis = 2)
        Y_ = np.concatenate((Ym, Ym, Ym), axis = 2)
    # tweak colors
    if np.random.rand() < 0.7:
        ch = int(np.random.rand() * 3)
        m = np.min((X_, Y_))
        offset = np.random.rand() * 0.25 - np.random.rand() * m
        X_[:, :, ch] = X_[:, :, ch] + offset * (1.0 - X_[:, :, ch])
        Y_[:, :, ch] = Y_[:, :, ch] + offset * (1.0 - Y_[:, :, ch])
    # flip channels
    if np.random.rand() < 0.7:
        seq = np.arange(3)
        np.random.shuffle(seq)
        Xtmp = np.copy(X_)
        Ytmp = np.copy(Y_)
        for i in range(3):
            X_[:, :, i] = Xtmp[:, :, seq[i]]
            Y_[:, :, i] = Ytmp[:, :, seq[i]]
    
    return (X_, Y_)

def get_train_samples(Xtr, Ytr, num = 1000, size = WINDOW_SIZE):
    assert(Xtr.shape[1] == Ytr.shape[1])
    X_ = np.zeros((num, size, size, 3), dtype = np.float32)
    Y_ = np.zeros((num, size, size, 3), dtype = np.float32)
    l = Xtr.shape[1]
    
    for i in range(num):
        ind = int(np.random.rand() * l)
        (X_[i], Y_[i]) = get_input_img(Xtr[ind], Ytr[ind], size)
    
    return (X_, Y_)

def get_train_samples_with_augmentation(Xtr, Ytr, num = 1000, size = WINDOW_SIZE):
    assert(len(Xtr) == len(Ytr))
    X_ = np.zeros((num, size, size, 3), dtype = np.float32)
    Y_ = np.zeros((num, size, size, 3), dtype = np.float32)
    l = len(Xtr)
    
    for i in range(num):
        ind = int(np.random.rand() * l)
        (X_[i], Y_[i]) = get_input_img_with_augmentation(Xtr[ind], Ytr[ind], size)
    
    return (X_ * 2 - 1, Y_ * 2 - 1)

def get_test_samples(Xtr, num = 1000, size = WINDOW_SIZE):
    X_ = np.zeros((num, size, size, 3), dtype = np.float32)

    for i in range(num):
        (X_[i], _) = get_input_img(Xtr, Xtr, size)
    
    return X_ * 2 - 1