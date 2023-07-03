import numpy as np
def set_significance(p):
    if p < 0.05:
        significance = '*'
    elif p < 0.005:
        significance = '**'
    else:
        significance = 'NA'
    return significance


def select_emotions(select_type=1):

    """
    :param select_type:select either normal emotions[1] or social emotions[2] or all[3]
    :return: indices for the emotions
    """

    EMOTION_LABELS_NORMAL = ['SURPRISED', 'EXCITED', 'SERENE', 'HAPPY', 'SATISFIED', 'CALM', 'TIRED', 'BORED', 'DEPRESSED', 'SAD', 'FRUSTRATED', 'AFRAID', 'ANGRY', 'STRESSED', 'ASTONISHED', 'SLEEPY', 'ALARMED', 'DISGUSTED']
    # Experiment 2
    EMOTION_LABELS_SOCIAL = ['SURPRISED', 'EXCITED', 'HAPPY', 'SERENE', 'SATISFIED', 'CALM', 'TIRED', 'BORED', 'PITY', 'SAD', 'FRUSTRATED', 'GENEROUS', 'ANGRY', 'ALARMED', 'AFRAID', 'CURIOUS', 'UNCERTAIN', 'DISGUSTED', 'JEALOUS', 'INTERESTED', 'SHY', 'EMPATHETIC', 'GUILTY']
    # Experiment 4
    EMOTION_LABELS_ALL = ['EXCITED', 'UNCERTAIN', 'SATISFIED', 'PITY', 'ANGRY', 'DEPRESSED', 'DISGUSTED', 'FRUSTRATED', 'JEALOUS', 'BORED', 'AFRAID', 'SURPRISED', 'STRESSED', 'TIRED', 'CURIOUS', 'SLEEPY', 'SERENE', 'ALARMED', 'GUILTY', 'GENEROUS', 'ASTONISHED', 'SAD', 'CALM', 'INTERESTED', 'SHY', 'HAPPY', 'EMPATHETIC']
    indices = []
    if select_type == 1:
        for i in range(0, len(EMOTION_LABELS_NORMAL)):
            for j in range(0, len(EMOTION_LABELS_ALL)):
                if EMOTION_LABELS_NORMAL[i] == EMOTION_LABELS_ALL[j]:
                    indices.append(j)
    elif select_type == 2:
        for i in range(0, len(EMOTION_LABELS_SOCIAL)):
            for j in range(0, len(EMOTION_LABELS_ALL)):
                if EMOTION_LABELS_SOCIAL[i]==EMOTION_LABELS_ALL[j]:
                    indices.append(j)
    else:
        indices = np.arange(len(EMOTION_LABELS_ALL))

    return np.asarray(indices)
