import numpy as np
import sklearn.metrics.cluster as cluster_metrics
from scipy.io import loadmat
from matplotlib.pyplot import *
from sklearn.cluster import k_means
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

def loadData():
    """Gets the data from a .mat file.
        X, y, attributeNames
    """

    mat_data = loadmat('./KidneyData.mat')
    attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
    X = mat_data['X']
    y = mat_data['y']

    return X, y, attributeNames
    
def clusterplot(X, clusterid, centroids='None', y='None', covars='None'):
    '''
    CLUSTERPLOT Plots a clustering of a data set as well as the true class
    labels. If data is more than 2-dimensional it should be first projected
    onto the first two principal components. Data objects are plotted as a dot
    with a circle around. The color of the dot indicates the true class,
    and the cicle indicates the cluster index. Optionally, the centroids are
    plotted as filled-star markers, and ellipsoids corresponding to covariance
    matrices (e.g. for gaussian mixture models).

    Usage:
    clusterplot(X, clusterid)
    clusterplot(X, clusterid, centroids=c_matrix, y=y_matrix)
    clusterplot(X, clusterid, centroids=c_matrix, y=y_matrix, covars=c_tensor)
    
    Input:
    X           N-by-M data matrix (N data objects with M attributes)
    clusterid   N-by-1 vector of cluster indices
    centroids   K-by-M matrix of cluster centroids (optional)
    y           N-by-1 vector of true class labels (optional)
    covars      M-by-M-by-K tensor of covariance matrices (optional)
    '''
    
    X = np.asarray(X)
    cls = np.asarray(clusterid)
    if type(y) is str and y=='None':
        y = np.zeros((X.shape[0],1))
    else:
        y = np.asarray(y)
    if type(centroids) is not str:
        centroids = np.asarray(centroids)
    K = np.size(np.unique(cls))
    C = np.size(np.unique(y))
    ncolors = np.max([C,K])
    
    # plot data points color-coded by class, cluster markers and centroids
    #hold(True)
    colors = [0]*ncolors
    for color in range(ncolors):
        colors[color] = cm.jet(color/(ncolors-1))[:3]
    for i,cs in enumerate(np.unique(y)):
        plot(X[(y==cs).ravel(),0], X[(y==cs).ravel(),1], 'o', markeredgecolor='k', markerfacecolor=colors[i],markersize=6, zorder=2)
    for i,cr in enumerate(np.unique(cls)):
        plot(X[(cls==cr).ravel(),0], X[(cls==cr).ravel(),1], 'o', markersize=12, markeredgecolor=colors[i], markerfacecolor='None', markeredgewidth=3, zorder=1)
    if type(centroids) is not str:        
        for cd in range(centroids.shape[0]):
            plot(centroids[cd,0], centroids[cd,1], '*', markersize=22, markeredgecolor='k', markerfacecolor=colors[cd], markeredgewidth=2, zorder=3)
    # plot cluster shapes:
    if type(covars) is not str:
        for cd in range(centroids.shape[0]):
            x1, x2 = gauss_2d(centroids[cd],covars[cd,:,:])
            plot(x1,x2,'-', color=colors[cd], linewidth=3, zorder=5)
    #hold(False)

    # create legend        
    legend_items = np.unique(y).tolist()+np.unique(cls).tolist()+np.unique(cls).tolist()
    for i in range(len(legend_items)):
        if i<C: legend_items[i] = 'Class: {0}'.format(legend_items[i]);
        elif i<C+K: legend_items[i] = 'Cluster: {0}'.format(legend_items[i]);
        else: legend_items[i] = 'Centroid: {0}'.format(legend_items[i]);
    legend(legend_items, numpoints=1, markerscale=.75, prop={'size': 9})

def gauss_2d(centroid, ccov, std=2, points=100):
    ''' Returns two vectors representing slice through gaussian, cut at given standard deviation. '''
    mean = np.c_[centroid]; tt = np.c_[np.linspace(0, 2*np.pi, points)]
    x = np.cos(tt); y=np.sin(tt); ap = np.concatenate((x,y), axis=1).T
    d, v = np.linalg.eig(ccov); d = std * np.sqrt(np.diag(d))
    bp = np.dot(v, np.dot(d, ap)) + np.tile(mean, (1, ap.shape[1])) 
    return bp[0,:], bp[1,:]

def clusterval(y, clusterid):
    '''
    CLUSTERVAL Estimate cluster validity using Entropy, Purity, Rand Statistic,
    and Jaccard coefficient.
    
    Usage:
      Entropy, Purity, Rand, Jaccard = clusterval(y, clusterid);
    
    Input:
       y         N-by-1 vector of class labels 
       clusterid N-by-1 vector of cluster indices
    
    Output:
      Entropy    Entropy measure.
      Purity     Purity measure.
      Rand       Rand index.
      Jaccard    Jaccard coefficient.
    '''
    NMI = cluster_metrics.supervised.normalized_mutual_info_score(y,clusterid)
    
    #y = np.asarray(y).ravel(); clusterid = np.asarray(clusterid).ravel()
    C = np.unique(y).size; K = np.unique(clusterid).size; N = y.shape[0]
    EPS = 2.22e-16
    
    p_ij = np.zeros((K,C))          # probability that member of i'th cluster belongs to j'th class
    m_i = np.zeros((K,1))           # total number of objects in i'th cluster
    for k in range(K):
        m_i[k] = (clusterid==k).sum()
        yk = y[clusterid==k]
        for c in range(C):
            m_ij = (yk==c).sum()    # number of objects of j'th class in i'th cluster
            p_ij[k,c] = m_ij.astype(float)/m_i[k]
    entropy = ( (1-(p_ij*np.log2(p_ij+EPS)).sum(axis=1))*m_i.T ).sum() / (N*K) 
    purity = ( p_ij.max(axis=1) ).sum() / K

    f00=0; f01=0; f10=0; f11=0
    for i in range(N):
        for j in range(i):
            if y[i]!=y[j] and clusterid[i]!=clusterid[j]: f00 += 1;     # different class, different cluster    
            elif y[i]==y[j] and clusterid[i]==clusterid[j]: f11 += 1;   # same class, same cluster
            elif y[i]==y[j] and clusterid[i]!=clusterid[j]: f10 += 1;   # same class, different cluster    
            else: f01 +=1;                                              # different class, same cluster
    rand = np.float(f00+f11)/(f00+f01+f10+f11)
    jaccard = np.float(f11)/(f01+f10+f11)

    return rand, jaccard, NMI

def simpleClustering(X, y):
    # Number of clusters:
    K = 4

    # K-means clustering:
    centroids, cls, inertia = k_means(X,K)
        
    # Plot results:
    figure(figsize=(14,9))
    clusterplot(X, cls, centroids, y)
    show()

def hierarchicalClustering(X, y):
    '''Displays a clustered plot for a Data Set'''
    # Perform hierarchical/agglomerative clustering on data matrix
    Method = 'single'
    Metric = 'euclidean'

    Z = linkage(X, method=Method, metric=Metric)

    # Compute and display clusters by thresholding the dendrogram
    Maxclust = 4
    cls = fcluster(Z, criterion='maxclust', t=Maxclust)
    figure(1)
    clusterplot(X, cls.reshape(cls.shape[0],1), y=y)

    # Display dendrogram
    max_display_levels=6
    figure(2,figsize=(10,4))
    dendrogram(Z, truncate_mode='level', p=max_display_levels)

    show()

def clusterValidity(X, y):
    # Maximum number of clusters:
    K = 10

    # Allocate variables:
    Rand = np.zeros((K,))
    Jaccard = np.zeros((K,))
    NMI = np.zeros((K,))

    for k in range(K):
        # run K-means clustering:
        #cls = Pycluster.kcluster(X,k+1)[0]
        centroids, cls, inertia = k_means(X,k+1)
        # compute cluster validities:
        Rand[k], Jaccard[k], NMI[k] = clusterval(y,cls)    
            
    # Plot results:

    figure(1)
    title('Cluster validity')
    plot(np.arange(K)+1, Rand)
    plot(np.arange(K)+1, Jaccard)
    plot(np.arange(K)+1, NMI)
    ylim(-2,1.1)
    legend(['Rand', 'Jaccard', 'NMI'], loc=4)
    show()

def complexKClustering(X, y):
    N, M = X.shape
    # Image resolution and number of colors
    x = 40 
    y = 40
    c = 3


    # Number of clusters:
    K = 10

    # Number of repetitions with different initial centroid seeds
    S = 1

    # Run k-means clustering:
    centroids, cls, inertia = k_means(X, K, verbose=True, max_iter=100, n_init=S)


    # Plot results:

    # Plot centroids
    figure(1)
    n1 = np.ceil(np.sqrt(K/2)); n2 = np.ceil(np.float(K)/n1);
    for k in range(K):
        subplot(n1,n2,k+1)
        imshow(np.reshape(centroids[k,:],(c,x,y)).T,interpolation='None',cmap=cm.binary)
        xticks([]); yticks([])
        if k==np.floor((n2-1)/2): title('Centroids')

    # Plot few randomly selected faces and their nearest centroids    
    L = 5       # number of images to plot
    j = np.random.randint(0, N, L)
    figure(2)
    for l in range(L):
        subplot(2,L,l+1)
        imshow(np.resize(X[j[l],:],(c,x,y)).T,interpolation='None',cmap=cm.binary)
        xticks([]); yticks([])
        if l==np.floor((L-1)/2): title('Randomly selected faces and their centroids')
        subplot(2,L,L+l+1)
        imshow(np.resize(centroids[cls[j[l]],:],(c,x,y)).T,interpolation='None',cmap=cm.binary)
        xticks([]); yticks([])

    show()
