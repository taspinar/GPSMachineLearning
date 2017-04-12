import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import zscore
from util_cluster import some_cluster

def start():
    print(os.getcwd())
    data_folder = '../processed_data/'
    OUTPUT_FOLDER = '../processed_data/'

    X = np.loadtxt(data_folder+'all_meta_X.csv',delimiter=',',dtype=np.float32)
    y = np.loadtxt(data_folder+'all_meta_Y.csv',delimiter=',',dtype=np.int16)

    """Start on visualization"""
    N,D = X.shape
    mean = np.mean(X,0)
    std = np.std(X,0)
    mask = []
    for i,x in enumerate(X):
         if np.isnan(np.sum(x)):
             continue
         elif np.any(np.abs((x-mean)/std)>4.0):
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
    X,y = start()
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

    #tSNE
    for per in [150, 200, 300]:
        print('tSNE on perplexity %i'%per)
        X_red, y_red = tSNE(X, y, per)
        plt.figure()
        y_color, y_s = some_cluster(y_red)
        plt.scatter(X_red[:, 0], X_red[:, 1], c=y_color, s=y_s, marker='o',
                    linewidths=0)  # , linewidths=0,c=MNIST_valid[1])
        plt.title('Mode=argmax')
        plt.savefig('tSNE%i.png'%per)
        plt.show(block=True)

      
          
    # f3, ax3 = plt.subplots(1, D+1)
#
#      for d in range(D):
#          ax3[d].scatter(X_red[:, 0], X_red[:, 1],c =y_red[:,d] ,marker='*', linewidths=0)  # , linewidths=0,c=MNIST_valid[1])
#          ax3[d].set_title('Mode%3i'%d)
#          plt.savefig('myfig%s%i.png'%(fun.__name__,d))
#
#      plt.show(block=True)
