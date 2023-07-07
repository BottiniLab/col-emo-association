"""Core functions for the generation of figure X.

Author: Mattia Silvestri

This module contains the core functions for the performance of analsysis and the
generation of figures for the related to the NLP section of the paper.
"""

import importlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
from scipy import spatial
import itertools
from scipy.stats import zscore
from scipy.stats import spearmanr
from scipy import stats
import webcolors
from sklearn.metrics import pairwise
import statistics

# Add dependences
from module_nlp.utils import ce_nlp as ce
_ = importlib.reload(ce)

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None

def get_wheels(config):
    """Compute Osgood dimensions using centroids
    
    Parameters
    ----------
    dotprod: bool
        Whether or not use the dotproduct with the difference vector to prject
        into osgoodspace
    
    """

    #--- Unpack config ---#

    data = pd.read_feather('..' + config['model'])
    valence_dict = config['valence']
    arousal_dict = config['arousal']
    color_words = config['color_words']
    save_coords = config['save_coords']
    neighbours = config['neighbours']

    #---------------------#

    model = ce.Embeddings(data)

    try:
        model.model.drop("level_0", axis=1, inplace=True)
    except:
        pass

    # Extract coloremotion words
    select_words = {'Colors': color_words,
                    'Emotions': ['surprised', 'excited', 'serene', 'happy', 'satisfied', 'calm', 'tired', 'bored', 'depressed', 'sad',
                                 'frustrated', 'afraid', 'angry', 'stressed', 'astonished', 'sleepy', 'alarmed', 'disgusted']}

    dimensions = ce.get_lowspace(model, neighbours, valence_dict, arousal_dict) 

    n_words = len(select_words['Colors']) + len(select_words['Emotions'])
    # Project color and emotion vectors onto the difference vectors
    all_words = [item for sub in list(select_words.values()) for item in sub]
    coords = pd.DataFrame(data={'index': all_words,
                                        'Valence': np.zeros(n_words), 
                                        'Arousal': np.zeros(n_words), 
                                        })
    for index, row in coords.iterrows():
        # Get vector for the current coloremtion word,
        # row[0] is the string with the word at the current row
        current_word = ce.get_row(model.model, row[0])
        # Compute dot product between current word vector in fasttext space and
        # difference vectors
        coords.loc[index, 'Valence'] = current_word @ \
        dimensions['Valence'][0]
        coords.loc[index, 'Arousal'] = current_word @ \
        dimensions['Arousal'][0]
        
        df_colors = coords.iloc[0:len(color_words),:]
        df_colors[['z_valence', 'z_arousal']] = df_colors[['Valence',
                                                           'Arousal']].astype(np.float64).apply(zscore, axis=0)
        df_colors['Condition'] = list(itertools.repeat('Color',
                                                    len(select_words['Colors'])))
        df_emotions = \
                coords.tail(len(select_words['Emotions']))
        df_emotions[['z_valence', 'z_arousal']] = df_emotions[['Valence',
                                                           'Arousal']].astype(np.float64).apply(zscore, axis=0)
        df_emotions['Condition'] = list(itertools.repeat('Emotion',
                                                    len(select_words['Emotions'])))
        
        final = pd.concat([df_colors, df_emotions], axis=0)
        # final.reset_index(inplace=True)

    #--- Prepare data for common plot (coloured emotions) ---#
    # Compute pairwise euclidean distance between each dot

    ce_distance = spatial.distance.cdist(df_emotions[['z_valence',
                                                      'z_arousal']].values.astype('float'),
                                         df_colors[['z_valence',
                                                    'z_arousal']].values.astype('float'), 
                                         metric='euclidean')

    distances = pd.DataFrame(1 / ce_distance, index=select_words['Emotions'], columns=select_words['Colors'])
    hex_colors = {'green': np.array([0,128,0]),
                'orange': np.array([255,165,0]),
                'yellow': np.array([255,255,0]),
                'blue': np.array([0,0,255]),
                'purple': np.array([128,0,128]),
                'red': np.array([255,0,0]),
                  }

    distances['hex_codes'] = ''

    n_colors = len(select_words['Colors'])
    for i in distances.index:
        sorted_row = distances[list(hex_colors.keys())].sort_values(i, axis
                                = 1, ascending = False).loc[[i]]
        close_colors = sorted_row.columns.values[0:2]
        hex_np = np.array([hex_colors.get(k) for k in close_colors])
        temp = np.empty((2,3))
        count = 0
        for j in close_colors:
            temp[count] = (distances[j][i] * hex_np[count]) # Weight the colors
            for a in range(len(temp[count])):
                if temp[count][a] > 255:
                    temp[count][a] = 255
            count += 1

        average = np.mean(temp, axis=0)
        mean_rgb = tuple(map(int, average))
        distances['hex_codes'][i] = webcolors.rgb_to_hex(mean_rgb)

    # Create a dictionary to associate every emotion with a color
    emotions_colors = {}
    for e in range(len(df_emotions.index)):
        current_emotion = df_emotions.index[e]
        close_distance = min(ce_distance[e])
        close_color_idx = list(np.where(ce_distance[e] == close_distance)[0])[0]
        emotions_colors[current_emotion] = df_colors.index[close_color_idx]

    # Add fill_colur column to help with scatterplot
    n_colors = len(color_words)
    final['fill_color'] = list(itertools.repeat(0, len(final.index)))
    for i in range(len(final.index)):
        if final.iloc[i]['Condition'] == 'Color':
            final['fill_color'][i] = final.iloc[i]['index']
        else:
            final['fill_color'][i] = distances['hex_codes'].iloc[i-n_colors]

    final.to_csv('..' + save_coords)

    return final


def control_rdm(config: dict) -> None: 
    '''Compute and save the RDMs for the control conditions:
    
    - fasttext
    - fasttext2d
    - controlspace
    
    '''
    #--- Read config ---#

    data = pd.read_feather('..' + config['model_path'])
    control_coords = pd.read_csv('..' + config['path2control'])
    color_words = config['color_words']
    emotion_words = config['emotion_words']
    save_control = config['save_control']

    #---------------------#
    
    select_words = {'Colors': color_words,
                    'Emotions': emotion_words}
    
    model = ce.Embeddings(data)

    try:
        model.model.drop('level_0', axis=1, inplace=True)
    except:
        pass

    # Do PCA
    fasttext_2d = ce.compute_pca(data)[0]
    model_2d = ce.Embeddings(fasttext_2d.iloc[:,0:3])
    for words in select_words:
        # Compute rdm based on cosine distance in the original 300D space
        embeddings = {k:model.get_vector(k) for k in select_words[words]}
        rdm_fasttext = pairwise.cosine_distances(list(embeddings.values()))
        rdm_fasttext_df =  pd.DataFrame(data=rdm_fasttext, index=select_words[words], columns=select_words[words])

        # Compute RDM based on euclidean distance between words in the reduced fasttext space
        embeddings_2d = {k:model_2d.get_vector(k) for k in select_words[words]}
        rdm_fasttext2d = pairwise.euclidean_distances(list(embeddings_2d.values()))
        rdm_fasttext2d_df = pd.DataFrame(data=rdm_fasttext2d, index=select_words[words], columns=select_words[words])

        # Compute RDM based on euclidean distance in the control space
        ## Select words
        control_coords.set_index('index', inplace=True)
        coords_geo = control_coords.loc[filter(lambda x : x in select_words[words],
                                                   control_coords.index.values)]
        control_coords.reset_index(inplace=True)
        coords_geo.reset_index(inplace=True)
        ## Order words
        coords_geo['ordering'] = list(itertools.repeat(0, len(select_words[words])))
        count = 0
        for i in select_words[words]:
            coords_geo['ordering'].loc[coords_geo['index'] == i] = \
            count
            count += 1
        coords_geo.sort_values(by='ordering', axis=0, inplace=True)
        coords_geo.drop('ordering', axis=1)
        # TEST: mer 10 mag 2023, 15:47:17
        # Test that the order is correct
        assert list(coords_geo['index']) == select_words[words]
        ## Compute controlspace RDM
        rdm_control = pairwise.euclidean_distances(coords_geo[['Valence',
                                                               'Arousal']].values)
        rdm_control_df = pd.DataFrame(data=rdm_control, index=select_words[words],
                                      columns=select_words[words])


        # Save the matrices
        rdm_fasttext_df.to_csv('..' + save_control + words + '_fasttext300.csv')
        rdm_fasttext2d_df.to_csv('..' + save_control + words + '_fasttext2d.csv')
        rdm_control_df.to_csv('..' + save_control + words + '_controlspace.csv')
