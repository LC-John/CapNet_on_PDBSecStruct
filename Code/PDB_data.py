# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 19:51:22 2018

@author: DrLC
"""

import pandas as pd
import pickle, gzip
import numpy
import random

def read_cvs(path="../Dataset/ss_simp_center.csv"):
    
    data = pd.read_csv(path)
    data = (list(data['2']), list(data['3']))
    return data
    
def simp_gen(path="../Dataset/ss_simp_center.csv",
             final="../Dataset/ss_simp_centor.pkl.gz"):
    
    d = read_cvs(path)
    f = gzip.open(final, 'wb')
    pickle.dump(d, f)
    f.close()
    
def load_simp(path="../Dataset/ss_simp_centor.pkl.gz"):
    
    f = gzip.open(path, 'rb')
    d = pickle.load(f)
    f.close()
    return d
    
def simp_str2vec(data,
                 AA_dict={'/':0, 'A':1, 'C':2, 'D':3,
                          'E':4, 'F':5, 'G':6, 'H':7,
                          'I':8, 'K':9, 'L':10,'M':11,
                          'N':12,'P':13,'Q':14,'R':15,
                          'S':16,'T':17,'V':18,'W':19,
                          'Y':20,'X':21},
                 label_dict={'L':0,'H':1,'E':2}):
    
    data_ = ([], [])
    for seq in data[0]:
        seq_ = []
        for aa in seq:
            seq_.append([0 for i in range(len(AA_dict))])
            seq_[-1][AA_dict[aa]] = 1
        data_[0].append(seq_)
    for l in data[1]:
        data_[1].append(label_dict[l])
        
    return data_
    
    
class Dataset():
    
    def __init__(self, X, Y, seqlen=21, rand_seed=1234, 
                 embedding={'/':0, 'A':1, 'C':2, 'D':3,
                            'E':4, 'F':5, 'G':6, 'H':7,
                            'I':8, 'K':9, 'L':10,'M':11,
                            'N':12,'P':13,'Q':14,'R':15,
                            'S':16,'T':17,'V':18,'W':19,
                            'Y':20,'X':21},
                 label={'L':0,'H':1,'E':2}):
        
        _X = numpy.copy(X)
        _Y = numpy.copy(Y)
        
        self.X = _X
        self.Y = _Y
        self.seqlen = seqlen
        self.emlen = len(embedding)
        self.__available = random.sample(range(self.Y.shape[0]),
                                         self.Y.shape[0])
        self.__embedding = embedding
        self.__label = label
        random.seed(rand_seed)
        numpy.random.seed(rand_seed)
    
    def minibatch(self, batchsize):
        
        if len(self.__available) < batchsize:
            self.__available = random.sample(range(self.Y.shape[0]),
                                             self.Y.shape[0])
        idx = self.__available[:batchsize]
        self.__available = self.__available[batchsize:]
        
        _X = [self.X[i] for i in idx]
        _Y = [self.Y[i] for i in idx]
        X = []
        Y = []
        for seq in _X:
            X.append([[0 for j in range(self.emlen)] for i in range(self.seqlen)])
            for n in range(len(seq)):
                X[-1][n][self.__embedding[seq[n]]] += 1
        for l in _Y:
            Y.append([0 for i in range(len(self.__label))])
            Y[-1][self.__label[l]] = 1
            
        X = numpy.asarray(X, dtype='float32')
        Y = numpy.asarray(Y, dtype='float32')

        return (X, Y)
    
    
if __name__ == "__main__":
    
    #simp_gen()
    d = load_simp("../Dataset/ss_orig_center.pkl.gz")
    D = Dataset(d[0], d[1], label={'C':0,'H':1,'G':2,'T':3,
                                   'S':4,'E':5,'B':6,'I':7})
    #d = load_simp()
    #D = Dataset(d[0], d[1])
    b = D.minibatch(32)
    print (b[0].shape)
    print (b[1].shape)