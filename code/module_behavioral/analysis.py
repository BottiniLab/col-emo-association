import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import kendalltau
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.manifold import MDS
from scipy.spatial.distance import euclidean
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import circle_fit as cf
from tqdm import tqdm

def reduce_dimensions(matrix, dims=2):
    pca = PCA(n_components=dims)
    embedding = pca.fit_transform(matrix)

    return embedding






