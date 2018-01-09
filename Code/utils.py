import os
import scipy
import numpy as np
import tensorflow as tf

import dataset
import PDB_data

def load_rna(batch_size, is_training=True):
    
    X, Y = dataset.load_data()
    D = dataset.Dataset(X, Y)
    X, Y = D.minibatch(len(X))
    X = X.reshape([2711, 60, 4, 1])
    Y = np.asarray(Y.argmax(axis=1), dtype=np.int32)
    
    if is_training:
        trX = X[:2200]
        trY = Y[:2200]

        valX = X[2200:,]
        valY = Y[2200:]

        num_tr_batch = 2200 // batch_size
        num_val_batch = 511 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
        
    else:
        return X, Y, 2711//batch_size
        
def load_pdb(batch_size, is_training=True):
    
    X, Y = PDB_data.load_simp()
    D = PDB_data.Dataset(X, Y)
    X, Y = D.minibatch(len(X))
    X = X.reshape([121359, 21, 22, 1])
    Y = np.asarray(Y.argmax(axis=1), dtype=np.int32)
    
    if is_training:
        trX = X[:100000]
        trY = Y[:100000]

        valX = X[100000:,]
        valY = Y[100000:]

        num_tr_batch = 100000 // batch_size
        num_val_batch = 21359 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
        
    else:
        return X, Y, 121359//batch_size
        
def load_ori_pdb(batch_size, is_training=True):
    
    X, Y = PDB_data.load_simp("../Dataset/ss_orig_center.pkl.gz")
    D = PDB_data.Dataset(X, Y, label={'C':0,'H':1,'G':2,'T':3,
                                      'S':4,'E':5,'B':6,'I':7})
    X, Y = D.minibatch(len(X))
    X = X.reshape([121359, 21, 22, 1])
    Y = np.asarray(Y.argmax(axis=1), dtype=np.int32)
    
    if is_training:
        trX = X[:100000]
        trY = Y[:100000]

        valX = X[100000:,]
        valY = Y[100000:]

        num_tr_batch = 100000 // batch_size
        num_val_batch = 21359 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
        
    else:
        return X, Y, 121359//batch_size
        
def load_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_fashion_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'fashion-mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_data(dataset, batch_size, is_training=True, one_hot=False):
    if dataset == 'mnist':
        return load_mnist(batch_size, is_training)
    elif dataset == 'fashion-mnist':
        return load_fashion_mnist(batch_size, is_training)
    elif dataset == "rna":
        return load_rna(batch_size, is_training)
    elif dataset == "pdb":
        return load_pdb(batch_size, is_training)
    elif dataset == "ori-pdb":
        return load_ori_pdb(batch_size, is_training)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def get_batch_data(dataset, batch_size, num_threads):
    if dataset == 'mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_mnist(batch_size, is_training=True)
    elif dataset == 'fashion-mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_fashion_mnist(batch_size, is_training=True)
    elif dataset == "rna":
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_rna(batch_size, is_training=True)
    elif dataset == "pdb":
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_pdb(batch_size, is_training=True)
    elif dataset == "ori-pdb":
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_ori_pdb(batch_size, is_training=True)
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)

    return(X, Y)


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs
