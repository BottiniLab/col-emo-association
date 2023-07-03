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


def emotion_positions_based_on_associations(color_embedding, average_emotion_color_ranking_matrix):

    N = average_emotion_color_ranking_matrix.shape[0] # set number of emotions
    emotion_colors_based_on_positions = np.zeros((N, 2)) # since we only do two dimensional PCA

    for i in range(0, average_emotion_color_ranking_matrix.shape[0]):
        x = 0
        y = 0
        for j in range(0, average_emotion_color_ranking_matrix[i, :].shape[0]):
            x = color_embedding[j, 0]*average_emotion_color_ranking_matrix[i, j]+x
            y = color_embedding[j, 1]*average_emotion_color_ranking_matrix[i, j]+y
        emotion_colors_based_on_positions[i, 0] = x
        emotion_colors_based_on_positions[i, 1] = y

    return emotion_colors_based_on_positions

def emotion_colors(color_embedding, color_ranking_matrix, rgb_values):

    emotion_colors_based_on_positions = emotion_positions_based_on_associations(color_embedding, color_ranking_matrix)
    total_emotions = len(color_ranking_matrix)

    distancesColorContributions = pairwise_distances(color_embedding, emotion_colors_based_on_positions)
    colorEmotionProfile = distancesColorContributions.T # transpose to make things easier.

    # for each emotion, we pick the two colors closest to it and normalise it based on the distance.

    averageColors = np.zeros((total_emotions, 3))
    for idx in range(0, colorEmotionProfile.shape[0]):
        row = colorEmotionProfile[idx, :]
        x = np.argsort(row)

        #normalize the two colors by distance, so that the one closer gets higher weightage
        contrib = row[x[0:2]]
        contrib = contrib/contrib.min()
        contrib = contrib/contrib.sum()
        contrib = 1-contrib
        y = np.zeros((1, 3))
        for j in range(0, len(contrib)):
            for i in range(0, 3):
                y[0, i] = y[0, i]+contrib[j]*rgb_values[x[j], i]
        averageColors[idx] = y

    #set maximum to one
    averageColors[averageColors>1] = 1

    return averageColors


def distance_from_centre(center_coordinate, coordinates):

    distances = []
    for c in coordinates:
        distances.append(np.linalg.norm(c-center_coordinate))
    distances = np.asarray(distances)
    return distances

def average_emotion_color(color_ranking_matrix, rgb_values):

    averageRGBColors = np.zeros((color_ranking_matrix.shape[0],3))
    for i in range(0, color_ranking_matrix.shape[0]):
        x = 0
        y = 0
        z = 0
        for j in range(0, color_ranking_matrix[i, :].shape[0]):
            x = rgb_values[j, 0]*color_ranking_matrix[i, j]+x
            y = rgb_values[j, 1]*color_ranking_matrix[i, j]+y
            z = rgb_values[j, 2]*color_ranking_matrix[i, j]+z

        averageRGBColors[i, 0] = x
        averageRGBColors[i, 1] = y
        averageRGBColors[i, 2] = z

    return averageRGBColors
def circle_fitting_and_regression(emotion_embedding, color_embedding, color_ranking_matrix, rgb_values):

    center_x, center_y, radius, error = cf.hyper_fit(emotion_embedding)
    center_emotion = np.asarray([center_x, center_y])

    #calculate distance of emotions from emotion center
    emotion_center_distances = distance_from_centre(center_emotion, emotion_embedding)

    #calculate distance of emotion positions weighed by color choices from the center of color embedding
    center_color = np.mean(color_embedding, axis=0)
    emotion_colors_based_on_positions = emotion_positions_based_on_associations(color_embedding, color_ranking_matrix)
    color_center_distances = distance_from_centre(center_color, emotion_colors_based_on_positions)

    #calculate distance of average emotion color choices from grey
    grey = np.asarray([128, 128, 128])/255
    emotions_colors_average = average_emotion_color(color_ranking_matrix, rgb_values)
    grey_center_distance = distance_from_centre(grey, emotions_colors_average)

    return center_emotion, radius, emotion_center_distances, color_center_distances, grey_center_distance









