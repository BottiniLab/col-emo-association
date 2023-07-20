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
import plotly.express as px
import os 

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
    emotion_words = config['emotion_words']
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
                    'Emotions': emotion_words}

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

    en_colors = ["green", "orange", "yellow", "blue", "purple", "red"]
    distances = pd.DataFrame(1 / ce_distance, index=select_words['Emotions'], columns=en_colors)
    rgb_colors = {'green': np.array([0,128,0]),
                'orange': np.array([255,165,0]),
                'yellow': np.array([255,255,0]),
                'blue': np.array([0,0,255]),
                'purple': np.array([128,0,128]),
                'red': np.array([255,0,0]),
                  }

    distances['hex_codes'] = ''

    n_colors = len(select_words['Colors'])
    for i in distances.index:
        sorted_row = distances[list(rgb_colors.keys())].sort_values(i, axis
                                = 1, ascending = False).loc[[i]]
        close_colors = sorted_row.columns.values[0:2]
        hex_np = np.array([rgb_colors.get(k) for k in close_colors])
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
            final['fill_color'][i] = en_colors[i]
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
        if not os.path.exists('..' + save_control):
            os.makedirs('..' + save_control)

        rdm_fasttext_df.to_csv('..' + save_control + words + '_fasttext300.csv')
        rdm_fasttext2d_df.to_csv('..' + save_control + words + '_fasttext2d.csv')
        rdm_control_df.to_csv('..' + save_control + words + '_controlspace.csv')


def osgood_rdm(config) -> None:
    """Same analysis as the previous function but in the Osgood space"""

    #--- Read config ---#

    emotion_wheel = pd.read_csv('..' + config['path_em_wheel'])
    color_wheel = pd.read_csv('..' + config['path_col_wheel'])
    osgoodaxes_df = pd.read_csv('..' + config['path_centroids'])
    path2control_rdm = config['path2control_rdm']
    color_words = config['color_words']
    emotion_words = config['emotion_words']
    path_results = config['path_results']

    #---------------------#

    if not os.path.exists('..' + path_results):
        os.makedirs('..' + path_results)

    select_words = {'Colors': color_words,
                    'Emotions': emotion_words}
                               

    def stat_test(osgoodaxes_df, rdm_human1, rdm_osgoodspace1, rdm_fasttext1,
                  rdm_fasttext2d1, rdm_control1):

        # Remove upper half diagonal
        rdm_human = ce.del_upper(rdm_human1)
        rdm_osgoodspace= ce.del_upper(rdm_osgoodspace1)
        rdm_fasttext = ce.del_upper(rdm_fasttext1)
        rdm_fasttext2d = ce.del_upper(rdm_fasttext2d1)
        rdm_control = ce.del_upper(rdm_control1.values)

        # Correlate the 3 RSM
        statistic = lambda x,y : spearmanr(x,y).correlation

        # Compute permutation test to test significance difference between the two
        # correlations
        corr_osgoodspace = spearmanr(rdm_human, rdm_osgoodspace)
        corr_fasttext = spearmanr(rdm_human, rdm_fasttext)
        corr_fasttext2d = spearmanr(rdm_human, rdm_fasttext2d)
        corr_control = spearmanr(rdm_human,
                                 rdm_control)

        s = 10000
        permute_osgoodspace = np.zeros(s)
        permute_fasttext = np.zeros(s)
        permute_fasttext2d = np.zeros(s)
        permute_control = np.zeros(s)

        for p in range(s):
            np.random.seed(p)
            human_sample = np.random.permutation(rdm_human)
            permute_osgoodspace[p] = spearmanr(human_sample, 
                                               np.random.permutation(rdm_osgoodspace))[0]
            permute_fasttext[p] = spearmanr(human_sample, 
                                               np.random.permutation(rdm_fasttext))[0]
            permute_fasttext2d[p] = spearmanr(human_sample, 
                                               np.random.permutation(rdm_fasttext2d))[0]
            permute_control[p] = spearmanr(human_sample, 
                                               np.random.permutation(rdm_control))[0]


        # Get p-value
        ## Fasttext
        diff_null = permute_osgoodspace - \
        permute_fasttext
        diff_corr = corr_osgoodspace[0] - corr_fasttext[0]
        diff_corr_z = statistics.NormalDist(mu = np.mean(diff_null),
                              sigma=np.std(diff_null)).zscore(diff_corr)

        p_val = stats.norm.sf(abs(diff_corr_z))*2

        ## Fasttext2d
        diff_null2d = permute_osgoodspace- \
        permute_fasttext2d
        diff_corr2d = corr_osgoodspace[0] - corr_fasttext2d[0]
        diff_corr_z2d = statistics.NormalDist(mu = np.mean(diff_null2d),
                              sigma=np.std(diff_null2d)).zscore(diff_corr2d)

        p_val2d = stats.norm.sf(abs(diff_corr_z2d))*2

        ## Control space
        diff_nullcont = permute_osgoodspace- \
        permute_control
        diff_corrcont = corr_osgoodspace[0] - corr_control[0]
        diff_corr_zcont = statistics.NormalDist(mu = np.mean(diff_nullcont),
                              sigma=np.std(diff_nullcont)).zscore(diff_corrcont)

        p_valcont = stats.norm.sf(abs(diff_corr_zcont))*2


        corr_df = pd.DataFrame(data=[[corr_osgoodspace[0], corr_osgoodspace[1]],
                                    [corr_fasttext[0], corr_fasttext[1]],
                                    [corr_fasttext2d[0], corr_fasttext2d[1]],
                                    [corr_control[0], corr_control[1]]],
                                    index=['osgoodspace', 'fasttext',
                                           'fasttext2d', 'controlspace'],
                                    columns=['coefficient', 'p-value'])

        diff_df = pd.DataFrame(data=[[p_val, diff_corr],
                                     [p_val2d, diff_corr2d],
                                     [p_valcont, diff_corrcont]], 
                                     index=['fasttext', 'fasttext2d',
                                            'controlsapce'], 
                                     columns=['p_val', 'r_coef'])
        #--- Plot the 4 RSM ---#
        # labels = osgoodaxes_df['index'].values
        # title_x = 0.5
        # width = 800
        # height = 800
        # font=dict(
        #     size=15
        #     )
        #
        # fig1 = px.imshow(rdm_osgoodspace1,
        #             title='Osgood space',
        #             x=labels,
        #             y=labels,
        #             color_continuous_scale = 'RdBu',
        #             width = width,
        #             height = height
        #            ) 
        #
        # fig1.update_xaxes(tickangle= 30)
        # fig1.update_layout(
        #     title_x=title_x,
        #     font=font
        #     )
        #
        # fig2 = px.imshow(rdm_fasttext1,
        #             title='Fasttext space',
        #             x=labels,
        #             y=labels,
        #             color_continuous_scale = 'RdBu',
        #             width = width,
        #             height = height
        #            ) 
        #
        # fig2.update_xaxes(tickangle= 30)
        # fig2.update_layout(
        #     title_x=title_x,
        #     font=font
        #     )
        #
        # fig3 = px.imshow(rdm_fasttext2d1,
        #             title='Fasttext reduced space',
        #             x=labels,
        #             y=labels,
        #             color_continuous_scale = 'RdBu',
        #             width = width,
        #             height = height
        #            ) 
        #
        # fig3.update_xaxes(tickangle= 30)
        # fig3.update_layout(
        #     title_x=title_x,
        #     font=font
        #     )
        #
        # fig4 = px.imshow(rdm_control1,
        #             title='Control space',
        #             x=labels,
        #             y=labels,
        #             color_continuous_scale = 'RdBu',
        #             width = width,
        #             height = height
        #            ) 
        #
        # fig4.update_xaxes(tickangle= 30)
        # fig4.update_layout(
        #     title_x=title_x,
        #     font=font
        #     )
        #
        # fig5 = px.imshow(rdm_human1,
        #             title='Human ratings',
        #             x=labels,
        #             y=labels,
        #             color_continuous_scale = 'RdBu',
        #             width = width,
        #             height = height
        #            ) 
        #
        # fig5.update_xaxes(tickangle= 30)
        # fig5.update_layout(
        #     title_x=title_x,
        #     font=font
        #     )

        # Save images to disk
        # fig1.write_image("../figures/supp1A.svg", format="svg", engine="kaleido")
        # fig2.write_image("../figures/supp1B.svg", format="svg", engine="kaleido")
        # fig3.write_image("../figures/supp1C.svg", format="svg", engine="kaleido")
        # fig4.write_image("../figures/supp1D.svg", format="svg", engine="kaleido")
        # fig5.write_image("../figures/supp1E.svg", format="svg", engine="kaleido")

        #--- Plot correlation scatterplot ---#
        osgood_flatten = pd.DataFrame({"Human ratings": rdm_human.flatten(),
                                       "Osgoodspace": rdm_osgoodspace.flatten()})
        fasttext_flatten = pd.DataFrame({"Human ratings": rdm_human.flatten(),
                                       "Fasttext": rdm_fasttext.flatten()})
        fasttext2d_flatten = pd.DataFrame({"Human ratings": rdm_human.flatten(),
                                       "Fasttext2D": rdm_fasttext2d.flatten()})
        control_flatten = pd.DataFrame({"Human ratings": rdm_human.flatten(),
                                       "Controlspace": rdm_control.flatten()})
        
        sns.set(font_scale=1.5)
        sns.set_style("white")
        sns.lmplot(x="Human ratings", y="Osgoodspace", data=osgood_flatten,
                   height=7, aspect=1.2, robust=True, palette="tab10",
                   scatter_kws=dict(s=60, linewidths=.7, edgecolors="black"))
        plt.title("Human ratings ~ Osgoodspace")
        plt.suptitle("r =  " + str(round(corr_osgoodspace[0], 3)), y=0.92)
        plt.xlabel("Human ratings (euclidean distance)")
        plt.ylabel("Osgood space (euclidean distance)")
        plt.savefig("../figures/supp1F.svg")
        plt.close()
        sns.lmplot(x="Human ratings", y="Fasttext", data=fasttext_flatten,
                   height=7, aspect=1.2, robust=True, palette="tab10",
                   scatter_kws=dict(s=60, linewidths=.7, edgecolors="black"))
        plt.title("Human ratings ~ Fasttext space" )
        plt.suptitle("r =  " + str(round(corr_fasttext[0], 3)), y=0.92)
        plt.xlabel("Human ratings (euclidean distance)")
        plt.ylabel("Fasttext space (cosine distance)")
        plt.savefig("../figures/supp1G.svg")
        plt.close()
        sns.lmplot(x="Human ratings", y="Fasttext2D", data=fasttext2d_flatten,
                   height=7, aspect=1.2, robust=True, palette="tab10",
                   scatter_kws=dict(s=60, linewidths=.7, edgecolors="black"))
        plt.title("Human ratings ~ Fasttext 2D space")
        plt.suptitle("r =  " + str(round(corr_fasttext2d[0], 3)), y=0.92)
        plt.xlabel("Human ratings (euclidean distance)")
        plt.ylabel("Fasttext 2D space (euclidean distance)")
        plt.savefig("../figures/supp1H.svg")
        plt.close()
        sns.lmplot(x="Human ratings", y="Controlspace", data=control_flatten,
                   height=7, aspect=1.2, robust=True, palette="tab10",
                   scatter_kws=dict(s=60, linewidths=.7, edgecolors="black"))
        plt.title("Human ratings ~ Control geographic space")
        plt.suptitle("r =  " + str(round(corr_control[0], 3)), y=0.92)
        plt.xlabel("Human ratings (euclidean distance)")
        plt.ylabel("Control space (euclidean distance)")
        plt.savefig("../figures/supp1I.svg")
        plt.close()

        return diff_df, corr_df


    for words in select_words:
        osgoodaxes_df.set_index('index', inplace=True)
        coords = osgoodaxes_df.loc[filter(lambda x : x in select_words[words],
                                                   osgoodaxes_df.index.values)]
        osgoodaxes_df.reset_index(inplace=True)
        coords.reset_index(inplace=True)
        # Make sure that words are ordered in the right way
        coords['ordering'] = list(itertools.repeat(0, len(select_words[words])))
        count = 0
        for i in select_words[words]:
            coords['ordering'].loc[coords['index'] == i] = count
            count += 1
        coords.sort_values(by='ordering', axis=0, inplace=True)
        coords.drop(['Unnamed: 0', 'ordering'], axis=1)

        # Compute Osgood space RSM
        rdm_osgoodspace = pairwise.euclidean_distances(coords[['Valence',
                                                                  'Arousal']].values)
        # Test for normality
        osgood_norm = ce.test_normality(ce.del_upper(rdm_osgoodspace), title='Osgood space')

        if words == 'Colors':
            # Compute human RDM
            rdm_human = pairwise.euclidean_distances(color_wheel[['x', 'y']].values)
            # Test for normality
            human_norm = ce.test_normality(ce.del_upper(rdm_human), title='Human ratings')

            # Load rdm based on cosine distance in the original 300D space
            rdm_fasttext = pd.read_csv('..' + path2control_rdm + 'Colors_fasttext300.csv')
            rdm_fasttext.set_index(rdm_fasttext.columns[0], inplace=True)
            # Test for normality
            fasttext_norm = ce.test_normality(ce.del_upper(rdm_fasttext.values), title='Fasttext 300D')

            # Load fasttext2d RDM
            rdm_fasttext2d = pd.read_csv('..' + path2control_rdm + 'Colors_fasttext2d.csv')
            rdm_fasttext2d.set_index(rdm_fasttext2d.columns[0], inplace=True)
            # Test for normality
            fasttext2d_norm = ce.test_normality(ce.del_upper(rdm_fasttext2d.values), title='Fasttext 2D')

            # Load controlspace RDM
            rdm_control = pd.read_csv('..' + path2control_rdm + 'Colors_controlspace.csv')
            rdm_control.set_index(rdm_control.columns[0], inplace=True)
            # Test for normality
            control_norm = ce.test_normality(ce.del_upper(rdm_control.values), title='Control space')

            corr_method = 'spearman'
            # Stats on color words
            diff_col, corr_col = stat_test(coords, rdm_human, rdm_osgoodspace, rdm_fasttext.values, rdm_fasttext2d.values, 
                rdm_control1=rdm_control)
            corr_col.to_csv('..' + path_results + 'colors_spearman1.csv')
            diff_col.to_csv('..' + path_results + 'colors_spearman2.csv')
        else: 
            # Compute human RDM
            rdm_human = pairwise.euclidean_distances(emotion_wheel[['x', 'y']].values)
            # Test for normality
            human_norm = ce.test_normality(ce.del_upper(rdm_human), title='Human ratings')

            # Load rdm based on cosine distance in the original 300D space
            rdm_fasttext = pd.read_csv('..' + path2control_rdm + 'Emotions_fasttext300.csv')
            rdm_fasttext.set_index(rdm_fasttext.columns[0], inplace=True)
            # Test for normality
            fasttext_norm = ce.test_normality(ce.del_upper(rdm_fasttext.values), title='Fasttext 300D')

            # Load fasttext2d RDM
            rdm_fasttext2d = pd.read_csv('..' + path2control_rdm + 'Emotions_fasttext2d.csv')
            rdm_fasttext2d.set_index(rdm_fasttext2d.columns[0], inplace=True)
            # Test for normality
            fasttext2d_norm = ce.test_normality(ce.del_upper(rdm_fasttext2d.values), title='Fasttext 2D')

            # Load controlspace RDM
            rdm_control = pd.read_csv('..' + path2control_rdm + 'Emotions_controlspace.csv')
            rdm_control.set_index(rdm_control.columns[0], inplace=True)
            # Test for normality
            control_norm = ce.test_normality(ce.del_upper(rdm_control.values), title='Control space')

            corr_method = 'spearman'
            diff_em, corr_em = stat_test(coords, rdm_human, rdm_osgoodspace, rdm_fasttext.values, rdm_fasttext2d.values, 
                rdm_control1=rdm_control)
            corr_em.to_csv('..' + path_results + 'emotions_spearman1.csv')
            diff_em.to_csv('..' + path_results + 'emotions_spearman2.csv')
