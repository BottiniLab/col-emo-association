"""Helper functions for the NLP section of the paper.

Author: Mattia Silvestri
"""
import pandas as pd
import numpy as np
import pickle

class Embeddings:
    """Class to handle word embeddings"""
    
    def __init__(self, model):
        self.model = model
    

    def get_vector(self, word = str) -> np.ndarray:
        """Get word vector"""
        df = self.model
        return df.loc[df['index'] == word].iloc[:,1:].values[0]


    def get_representatives(self, valence, arousal):
        '''Get embeddings for the representative words

        Inputs are dictionaries, each containig the two words representative for
        that dimension.

        '''

        valence_dict = {}    
        arousal_dict = {}
        dominance_dict = {}

        # Assign a vector to each words and save it as object attributes 
        for (a,b,c) in zip(valence, arousal):
            df = self.model
            valence_dict[a] = get_row(df, a)
            arousal_dict[b] = get_row(df, b)
            dominance_dict[c] = get_row(df, c)
        repr_dict = {
            'valence': valence_dict,
            'arousal': arousal_dict,
        }

        return repr_dict

    
    def diff(self, word1, word2):
        '''Compute normalized and non-normalized difference between two word vectors.
        
        Parameters
        ----------
        word1, word2: string
            The two words between which we want to compute the vectorial difference.

        Returns
        -------
        norm_diff: numpy.ndarray
            Unit vector resulting from the difference between the two words.

        v_diff: numpy.ndarray
            Vector resulting from the difference between the two words.
        
        '''

        vec1 = np.array(get_row(self.model, word1))
        vec2 = np.array(get_row(self.model, word2))
        v_diff = vec1 - vec2
        norm_diff = v_diff / np.linalg.norm(v_diff)

        return norm_diff, v_diff


    def get_neighbours(self, word=None, vector=[], k=20, no_go=None):
        '''Get neighbours of the word

        Computes cosine distance between the selected words and choose the k
        closest words.

        Parameters
        ----------
        word: str
            Compute neighbours of this word. If not provided, it assumes the
            input is a vector.

        vector: numpy.ndarray
            Compute neighbours of this vector. If not provided, it assumes the
            input is a word.

        no_go: list
            List of string holding the words that need to be removed from the
            neighbours.

        '''

        df = self.model
        # Compute cosine similarity between word and the rest of the words
        if vector == []:
            word_vec = get_row(df, word)
        else:
            word_vec = vector

        df['similarity'] = df.iloc[:,1:].apply(lambda word2 : spatial.distance.cosine(word_vec, word2), axis=1)

        df_sorted = df.sort_values('similarity', ascending=True)
        if no_go:
            for i in no_go:
                df_sorted.drop(df_sorted[df_sorted['index'] ==
                                             i].index.values, axis=0, inplace=True)

        df_sorted = df_sorted[~df_sorted['index'].str.contains(word + '\.')]
        neighbours_df = df_sorted.iloc[1:k+1,[0,-1]]
        neighbours = {k:v for k,v in zip(neighbours_df.iloc[:,0], neighbours_df.iloc[:,1])}
        self.model.drop('similarity', axis=1, inplace=True)

        return neighbours


def get_row(df, i):
    return df.loc[df['index'] == i].iloc[:,1:].values[0]


def get_lowspace(model:Embeddings, neighbours: str, valence: list, arousal:
                 list) -> dict:

    '''Get dimensions along which to plot the color and emotion words
    
    Parameters
    ----------
    model: Embeddings
        fastText model with embeddings of all the words
    neighbours: str
        Path to the file holding the neighbours used for the
        centroids.
    valence: list
        List with the two valence words.
    arousal: list 
        List with the two arousal words.

    Return
    ------
    difference_vectors: dict
        Dimensions along which to plot color and emotion words
    
    '''

    # Load 50 nearest neighbours to compute centroids
    with open(".." + neighbours, 'rb') as f:
        cloud = pickle.load(f)
    
    # Extract vectors for neighbouring words
    for v,a in zip(valence, arousal):
        # Add representative words vectors
        cloud['Valence'][v][v] = model.get_vector(v)
        cloud['Arousal'][a][a] = model.get_vector(a)
        for c,b in zip(cloud['Valence'][v], cloud['Arousal'][a]):
            # Add neighbours vectors
            cloud['Valence'][v][c] = model.get_vector(c)
            cloud['Arousal'][a][b] = model.get_vector(b)
            # TEST: Testing code -@mattia at 3/30/2023, 2:29:02 PM
            # Test that the cloud has the right amount of words
            assert len(cloud['Valence'][v]) == 51, \
            f"Valence cloud size mismatch: {len(cloud['Valence'][v][c])}"
            assert len(cloud['Arousal'][a]) == 51, \
            f"Arousal cloud size mismatch: {len(cloud['Arousal'][v][b])}"

    # Compute average vector over the 50 surrounding words + the representative
    # words
    centroids  = {'Valence': {valence[0]: [], valence[1]: []},
                  'Arousal': {arousal[0]: [], arousal[1]: []}}
    for k in centroids['Valence']:
        centroids['Valence'][k] = \
        np.mean(list(cloud['Valence'][k].values()), axis=0)
    for k in centroids['Arousal']:
        centroids['Arousal'][k] = \
        np.mean(list(cloud['Arousal'][k].values()), axis=0)

    # Compute difference between representative centroids
    difference_vectors = {'Valence': [], 'Arousal': []}
    for k in centroids:
        keys = list(centroids[k].keys())
        if "East" in keys:
            vec1 = centroids[k]["East"] 
            vec2 = centroids[k]["West"] 
            v_diff = vec1 - vec2
            norm_diff = v_diff / np.linalg.norm(v_diff)
            difference_vectors[k] = (norm_diff, v_diff)
        elif "Est" in keys:
            vec1 = centroids[k]["Est"] 
            vec2 = centroids[k]["Ovest"] 
            v_diff = vec1 - vec2
            norm_diff = v_diff / np.linalg.norm(v_diff)
            difference_vectors[k] = (norm_diff, v_diff)
        else:
            vec1 = centroids[k][keys[0]] # North
            vec2 = centroids[k][keys[1]] # South
            v_diff = vec1 - vec2
            norm_diff = v_diff / np.linalg.norm(v_diff)
            difference_vectors[k] = (norm_diff, v_diff)

    return difference_vectors
