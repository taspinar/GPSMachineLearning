import numpy as np

def softmax(x,t=1):
    t = float(t)
    xx = (x-np.max(x))/t
    ex = np.exp(xx)
    return ex/np.sum(ex,axis=0)

def rescale(x,srange):
    x -= np.min(x)
    x /= np.max(x)
    x *= srange[1]-srange[0]
    x += srange[0]
    return x


def some_cluster(y, colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k'],srange=(5,20)):
    # selects the color for some cluster on each row
    N,D = y.shape

    counts = [0]*(D+1)

    out = ['k']*N
    markersize = [0] * N
    for r,row in enumerate(y):
        if np.sum(row) > 0:
            c = np.random.choice(np.argwhere(row).flatten(),1)[0]
            counts[c] += 1
            out[r] = colors[c]
            markersize[r] = c
        else:
            counts[D] += 1
            markersize[r] = D

    p = softmax(np.array(counts),10000)
    s = rescale(1.0/p,srange)
    markersize = list(map(lambda x:s[x],markersize))
    return out,markersize

