# data_proc.py process the data into center/most form

import functions
import numpy as np
import pandas as pd
import re

df = pd.read_csv('./ss.csv')
s1 = list(df.loc[:, '0'])
s2 = list(df.loc[:, '1'])
s3 = list(df.loc[:, '2'])

s3 = functions.simplify_c(s3)
s3_simp = functions.simplify(s3)

ss_orig_center = functions.gen_center(s1, s2, s3, 500, 21)
ss_orig_most = functions.gen_most(s1, s2, s3, 2000, 21, 10)

ss_simp_center = functions.gen_center(s1, s2, s3_simp, 500, 21)
ss_simp_most = functions.gen_most(s1, s2, s3_simp, 2000, 21, 10)

ss_orig_center.to_csv('./ss_orig_center.csv')
ss_orig_most.to_csv('./ss_orig_most.csv')
ss_simp_center.to_csv('./ss_simp_center.csv')
ss_simp_most.to_csv('./ss_simp_most.csv')
