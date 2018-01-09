# data_trans.py transfer raw data into csv form

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

d = get_data('./ss.txt')

s1, s2, s3 = [], [], []
temp = []

s = d.pop(0)
d.append(s)

for line in d:
    if not re.search('>', line):
        temp.append(line)
    elif re.search('sequence', line):
        temp = ''.join(temp)
        s3.append(temp)
        temp = []
    elif re.search('secstr', line):
        temp = ''.join(temp)
        s2.append(temp)
        temp = []

        protid = re.sub(r':secstr', '', line)
        protid = re.sub(r'>', '', protid)
        s1.append(protid)

s = [s1, s2, s3]
df = pd.DataFrame(s)
df = df.transpose()
df.to_csv('./ss.csv')

