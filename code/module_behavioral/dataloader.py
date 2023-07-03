import numpy as np

class DataLoader:

    # define the constants that define matrix organisation for the loaded numpy datasets.
    # the indexes in the matrix follow the same order as the labels.
    # Experiment 1, 3, 5, 6
    EMOTION_LABELS_NORMAL = ['SURPRISED', 'EXCITED', 'SERENE', 'HAPPY', 'SATISFIED', 'CALM', 'TIRED', 'BORED', 'DEPRESSED', 'SAD', 'FRUSTRATED', 'AFRAID', 'ANGRY', 'STRESSED', 'ASTONISHED', 'SLEEPY', 'ALARMED', 'DISGUSTED']
    # Experiment 2
    EMOTION_LABELS_SOCIAL = ['SURPRISED', 'EXCITED', 'HAPPY', 'SERENE', 'SATISFIED', 'CALM', 'TIRED', 'BORED', 'PITY', 'SAD', 'FRUSTRATED', 'GENEROUS', 'ANGRY', 'ALARMED', 'AFRAID', 'CURIOUS', 'UNCERTAIN', 'DISGUSTED', 'JEALOUS', 'INTERESTED', 'SHY', 'EMPATHETIC', 'GUILTY']
    # Experiment 4
    EMOTION_LABELS_ALL = ['EXCITED', 'UNCERTAIN', 'SATISFIED', 'PITY', 'ANGRY', 'DEPRESSED', 'DISGUSTED', 'FRUSTRATED', 'JEALOUS', 'BORED', 'AFRAID', 'SURPRISED', 'STRESSED', 'TIRED', 'CURIOUS', 'SLEEPY', 'SERENE', 'ALARMED', 'GUILTY', 'GENEROUS', 'ASTONISHED', 'SAD', 'CALM', 'INTERESTED', 'SHY', 'HAPPY', 'EMPATHETIC']
    # Experiment 1, 2
    COLOR_LABELS_NORMAL = ["RED", "ORANGE", "YELLOW", "RGBGREEN", "GREEN", "LIMEGREEN", "CYAN", "LIGHTBLUE", "BLUE", "PURPLE", "MAGENTA", "PINK"]
    # Experiment 3, 5 ,6
    COLOR_LABELS_WORD = ['GREEN', 'ORANGE', 'YELLOW', 'BLUE', 'PURPLE', 'RED']


    # Color values
    RGB_VALUES_PATCHES = np.asarray([[255, 0, 0], [255, 128, 0], [255, 255, 0], [128, 255, 0], [0, 255, 0], [0, 255, 128], [0, 255, 255], [0, 128, 255], [0, 0, 255], [128, 0, 255], [255, 0, 255], [255, 0, 128]])/255
    RGB_VALUES_WORDS = np.asarray([[0, 255, 0], [255, 128, 0], [255, 255, 0],  [0, 0, 255], [128, 0, 255], [255, 0, 0]])/255

    def __init__(self, context, location):

        """
        :param context: sets which experiment is being analysed
        :param location: sets the location to load the files from
        """


        self.context = context
        self.location = location

        self.average_color_matrix = None
        self.individual_color_matrix = None

        self.average_emotion_matrix = None

        self.average_emotion_color_ranking_matrix = None
        self.individual_emotion_color_ranking_matrix = None

        self.average_emotion_color_rgb_matrix = None
        self.individual_emotion_color_rgb_matrix = None
        self.reported_confidence = None


        if self.context in [1, 3, 5, 6]:
            self.emotion = self.EMOTION_LABELS_NORMAL
        elif self.context == 2:
            self.emotion = self.EMOTION_LABELS_SOCIAL
        elif self.context == 4:
            self.emotion = self.EMOTION_LABELS_ALL
        else:
            raise Exception("Context not recognized.")

        if self.context in [1, 2]:
            self.color = self.COLOR_LABELS_NORMAL
            self.color_values = self.RGB_VALUES_PATCHES
        elif self.context in [3, 5, 6]:
            self.color = self.COLOR_LABELS_WORD
            self.color_values = self.RGB_VALUES_WORDS
        elif self.context == 4:
            self.color = 'NA'
            self.color_values = 'NA'
        else:
            raise Exception("Context not recognized.")

        self.number_of_colors = len(self.color)
        self.number_of_emotions = len(self.emotion)

    def load_color(self):

        data = np.load(self.location+'/participantsColor.npy', allow_pickle=True).item()

        #generate average matrix for color choices

        average_color_matrix = np.zeros((self.number_of_colors, self.number_of_colors))

        for participant in data.keys():
            average_color_matrix = average_color_matrix + data[participant]

        average_color_matrix = average_color_matrix / len(data.keys()) # averaging over the number of participants.

        self.average_color_matrix = (average_color_matrix + average_color_matrix.T)/14

        if self.context == 5:
            self.individual_color_matrix = data

    def load_emotion(self):

        data = np.load(self.location+'/participantsEmotion.npy', allow_pickle=True).item()

        #generate average matrix for emotion choices

        average_emotion_matrix = np.zeros((self.number_of_emotions, self.number_of_emotions))

        for participant in data.keys():
            average_emotion_matrix = average_emotion_matrix + data[participant]

        average_emotion_matrix = average_emotion_matrix / len(data.keys()) # averaging over the number of participants.

        self.average_emotion_matrix = (average_emotion_matrix+average_emotion_matrix.T)/14


    def load_emotion_color_association(self):

        data = np.load(self.location+'/participantsEmotionColorAssociation.npy', allow_pickle=True).item()

        # generate average matrix
        average_emotion_color_ranking_matrix = np.zeros((self.number_of_emotions, self.number_of_colors))

        for participant in data.keys():
            average_emotion_color_ranking_matrix = average_emotion_color_ranking_matrix + data[participant]

        average_emotion_color_ranking_matrix = average_emotion_color_ranking_matrix / len(data.keys())

        self.average_emotion_color_ranking_matrix = average_emotion_color_ranking_matrix
        if self.context == 5:
            self.individual_emotion_color_ranking_matrix = data

    def load_rgb_emotion_color_association(self):

        # shape -> participants x emotions x rgb values
        self.individual_emotion_color_rgb_matrix = np.load(self.location+'/allColorResponses.npy', allow_pickle=True)
        self.average_emotion_color_rgb_matrix = np.average(self.individual_emotion_color_rgb_matrix, axis=0)/255
        self.reported_confidence = np.load(self.location+'/participantReportedConfidence.npy', allow_pickle=True)
