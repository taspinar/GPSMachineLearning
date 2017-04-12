import os
import numpy as np
import pandas as pd

"""Get the directories right"""
print(os.getcwd())
data_folder = '../processed_data'
OUTPUT_FOLDER = '../processed_data/'

"""Define the one-hot encodings"""
hot =         {'walk': np.array([1, 0, 0, 0, 0, 0]),
               'train': np.array([0, 1, 0, 0, 0, 0]),
               'subway': np.array([0, 0, 1, 0, 0, 0]),
               'taxi': np.array([0, 0, 0, 1, 0, 0]),
               'bus': np.array([0, 0, 0, 0, 1, 0]),
               'bike': np.array([0, 0, 0, 0, 0, 1])}
D = len(hot['walk'])

"""Loop over the subfolders"""
subs = os.listdir(data_folder)
total_subs = len(subs)
total_lines = 0
with open(OUTPUT_FOLDER+'all_meta_X.csv','ab') as f_X, open(OUTPUT_FOLDER+'all_meta_Y.csv','ab') as f_Y:
    for i_d,d in enumerate(subs):
        if '_metadata' in d:
            df = pd.read_csv(os.path.join(data_folder, d))
            X = df.as_matrix(['v_ave','v_med','a_ave','a_med']).astype(np.float32)
            L = X.shape[0]
            total_lines += L
            Y = np.zeros((L,D),dtype=np.int16)

            labels = df.as_matrix(['labels'])
            modes = hot.keys()
            for il, l in enumerate(labels):
                try:
                    for m in l[0].split(','):
                        vec = hot.get(m,None)
                        if vec is not None:
                            Y[il] += vec
                except AttributeError:
                    pass

        np.savetxt(f_X, X, fmt='%10.3f', delimiter=',')
        np.savetxt(f_Y, Y, fmt='%3i', delimiter=',')
    print(total_subs)
    print(total_lines)
