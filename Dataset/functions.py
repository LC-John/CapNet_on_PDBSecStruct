# functions used in data_proc.py

import re
import pandas as pd
import numpy as np

def get_data(in_file):
    data = []
    for i in open(in_file, 'r', encoding='utf-8').readlines():
        if i != '\n':
            i = re.sub(r'\n', '', i)
            data.append(i)
    return data

def simplify_str(text):
    temp = re.sub(r'I', 'H', text)
    temp = re.sub(r'G', 'H', temp)
    temp = re.sub(r'B', 'E', temp)
    temp = re.sub(r'T', 'L', temp)
    temp = re.sub(r'S', 'L', temp)
    temp = re.sub(r'C', 'L', temp)
    temp = re.sub(r' ', 'L', temp)
    return temp

def simplify_cstr(text):
    temp = re.sub(r' ', 'C', text)
    return temp

def simplify(lis):
    temp = []
    for i in range(len(lis)):
        temp.append(simplify_str(lis[i]))
    return temp

def simplify_c(lis):
    temp = []
    for i in range(len(lis)):
        temp.append(simplify_cstr(lis[i]))
    return temp

def most(text):
    G = text.count('G')
    H = text.count('H')
    I = text.count('I')
    E = text.count('E')
    B = text.count('B')
    T = text.count('T')
    S = text.count('S')
    C = text.count('C')
    L = text.count('L')
    num = list([G,H,I,E,B,T,S,C,L])
    temp = 0
    ss = 'GHIEBTSCL'
    for i in range(9):
        if num[i] > temp:
            temp = num[i]
            mostss = ss[i]
        else:
            pass
    return mostss
  

def gen_most(name, protein, ss_, num_use, size, step):
    final = []
    for i in range(num_use):
        prot = protein[i]
        protid = name[i]
        ss = ss_[i]
        for j in range(0, len(prot)-size, step):
            temp_p = prot[j:j+size]
            temp_ss = most(ss[j:j+size])
            final.append([protid, j, temp_p, temp_ss])
    df = pd.DataFrame(final)
    return df

def center(protein, index, size):
    l = int((size-1)/2)
    s = ''
    for i in range(l):
        s = ''.join([s, '/'])
    protein = ''.join([s, protein, s])
    temp = protein[index:index+size]
    return temp

def gen_center(name, protein, ss_, num_use, size):
    final = []
    for i in range(num_use):
        protid = name[i]
        prot = protein[i]
        ss = ss_[i]
        for j in range(len(prot)):
            temp_p = center(prot, j, size)
            temp_ss = ss[j]
            final.append([protid, j, temp_p, temp_ss])
    df = pd.DataFrame(final)
    return df

