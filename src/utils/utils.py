import pandas as pd  
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from collections import OrderedDict



def dimension_reduction(df, n_components=2):
    data = TSNE(n_components=n_components, perplexity=40).fit_transform(df)
    return data

def clustering_kmeans(feature, n_cluster):
    n_clusters = n_cluster
    kmeans = KMeans(n_clusters, init='k-means++', n_init=10, max_iter=1000, algorithm='auto', verbose=0)
    kmeans.fit(df1)
    labels = kmeans.labels_
    return labels, kmeans


def cluster_dictionary(df1, labels):
   
    labels = labels.tolist()
    dict = {}

    for j in range(len(set(labels))):  # use set() to remove duplicates of list
        lis = []
        for i in range(len(labels)):
            # print("i", labels[i])
            if labels[i] == j:
                lis.append(df1.columns[i])
            else:
                pass
        dict[str(j)] = lis  # dictionary of clusters, key= label no and value = list of clusters
    
    return dict