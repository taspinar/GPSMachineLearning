import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import zscore
from util_cluster import some_cluster

#def start():
#    print(os.getcwd())
#    data_folder = '../processed_data/'
#    OUTPUT_FOLDER = '../processed_data/'
#
#    X = np.loadtxt(data_folder+'all_meta_X.csv',delimiter=',',dtype=np.float32)
#    y = np.loadtxt(data_folder+'all_meta_Y.csv',delimiter=',',dtype=np.int16)
#
#    """Start on visualization"""
#    N,D = X.shape
#    mean = np.mean(X,0)
#    std = np.std(X,0)
#    mask = []
#    for i,x in enumerate(X):
#         if np.isnan(np.sum(x)):
#             continue
#         elif np.any(np.abs((x-mean)/std)>4.0):
#             continue
#         else:
#             mask.append(i)
#    X = X[mask]
#    y = y[mask]
#    X = zscore(X)
#    return X,y

def munge():
    import glob
    import pandas as pd
    def clean_label(label):
        if not isinstance(label,float):
          return label.lstrip(',').rstrip(',').replace(',,', ',')

    INPUT_FOLDER = '../processed_data/'
    headers_metadf = ['trajectory_id', 'start_time', 'end_time', 'v_ave', 'v_med', 'v_max', 'a_ave', 'a_med', 'a_max',
                      'labels']

    list_df_metadata = []

    for file in glob.glob(INPUT_FOLDER + "*_metadata.csv"):
        df_metadata = pd.read_csv(file, index_col=0)
        list_df_metadata.append(df_metadata)

    
    df_metadata = pd.concat(list_df_metadata).dropna(subset=['v_ave', 'v_med', 'v_max', 'a_ave', 'a_med', 'a_max'])
    
    X = df_metadata.as_matrix(['v_ave', 'v_med', 'v_max', 'a_ave', 'a_med', 'a_max'])
    y = df_metadata['labels'].values

    N = X.shape[0]
    D = 6
    hot = {'walk': np.array([1, 0, 0, 0, 0, 0]),
           'train': np.array([0, 1, 0, 0, 0, 0]),
           'subway': np.array([0, 0, 1, 0, 0, 0]),
           'taxi': np.array([0, 0, 0, 1, 0, 0]),
           'bus': np.array([0, 0, 0, 0, 1, 0]),
           'bike': np.array([0, 0, 0, 0, 0, 1])}
    Y = np.zeros((N,D),dtype=np.int16)
    for iy in range(N):
        lbl = y[iy]
        if not isinstance(lbl, float):
            for key, value in hot.items():
                if key in lbl:
                    Y[iy] += value

    return X,Y

def remove_outliers(X,y):
    """Start on visualization"""
    N, D = X.shape
    mean = np.mean(X, 0)
    std = np.std(X, 0)
    mask = []
    for i, x in enumerate(X):
        if np.isnan(np.sum(x)):
            continue
        elif np.any(np.abs((x - mean) / std) > 2.0):
            continue
        else:
            mask.append(i)
    X = X[mask]
    y = y[mask]
    X = zscore(X)
    return X,y


def PCA(X,y):
    """PCA"""
    PCA_model = TruncatedSVD(n_components=2)
    reduced = PCA_model.fit_transform(X)
    return reduced,y

def tSNE(X,y,perp=30):
    #subsample to prevent memory error
    N = X.shape[0]
    ind = np.random.choice(N,10000)
    X = X[ind]
    y = y[ind]


    tSNE_model = TSNE(verbose=2,perplexity=perp,min_grad_norm=1E-07,n_iter=300,angle=0.6)
    reduced_tsne = tSNE_model.fit_transform(X)
    return reduced_tsne,y

if __name__ == '__main__':
    X, y = munge()
    X,y = remove_outliers(X,y)
    # X,y = start()
    N,D = y.shape
    
    funcs = [PCA,tSNE]
    colors = ['r', 'g', 'b', 'y','m','c','k']

    #PCA
    X_red,y_red = PCA(X,y)
    plt.figure()
    y_color,y_s = some_cluster(y_red)
    plt.scatter(X_red[:, 0], X_red[:, 1],c = y_color,s=y_s ,marker='o', linewidths=0)  # , linewidths=0,c=MNIST_valid[1])
    plt.title('Mode=argmax')
    plt.savefig('pca.png')
    plt.show(block=True)
    
    plt.figure()
    f4, ax4 = plt.subplots(1, D)
    for d in range(D):
        ax4[d].scatter(X_red[:, 0], X_red[:, 1],c =y_red[:,d] ,marker='*', linewidths=0)
        ax4[d].set_title('Mode%3i'%d)
    plt.setp([a.get_xticklabels() for a in ax4], visible=False)
    plt.setp([a.get_yticklabels() for a in ax4], visible=False)
    plt.savefig('pca_table.png')
    

    # #tSNE
    # for per in [300]:
    #     print('tSNE on perplexity %i'%per)
    #     X_red, y_red = tSNE(X, y, per)
    #     plt.figure()
    #     y_color, y_s = some_cluster(y_red)
    #     plt.scatter(X_red[:, 0], X_red[:, 1], c='k', s=y_s, marker='o',
    #                 linewidths=0)  # , linewidths=0,c=MNIST_valid[1])
    #     plt.title('Mode=argmax')
    #     plt.savefig('tSNE%i.png'%per)
    #     #
    #     plt.figure()
    #     f3, ax3 = plt.subplots(1, D)
    #     for d in range(D):
    #         ax3[d].scatter(X_red[:, 0], X_red[:, 1],c =y_red[:,d] ,marker='*', linewidths=0)
    #         ax3[d].set_title('Mode%3i'%d)
    #     plt.setp([a.get_xticklabels() for a in ax3], visible=False)
    #     plt.setp([a.get_yticklabels() for a in ax3], visible=False)
    #     plt.savefig('tsne_table.png')
