
# Experiment Details


For both the processed and the raw data, the experiment keys are the same.

| Key |                           Description                           |
|:---:|:---------------------------------------------------------------:|
|  1  |        Experiment with 12 color patches and 18 emotions         |
|  2  |        Experiment with 12 color patches and 23 emotions         | 
|  3  |          Experiment with 6 color words and 18 emotions          |
|  4  |    Participants were allowed to choose color on an RGB wheel    |
|  5  | Early Blind Italian participants, 6 color words and 18 emotions |
|  6  |       Italian participants, 6 color words and 18 emotions       |



### Data structure

The files in the **rawdata** folder are self-explanatory. 

- *colorsimilarity.csv*
    - for each participant in the experiment
    - it stores the response for each color pair.
- *emotionsimilarity.csv*
    - for each participant
    - it similarly stores the response for each emotion pair.
- *emotion_color_association.csv*
    - for each participant
    - it stores the color the participant associated the emotion with.
---

The files in the **processeddata** folder follow the following structure:

- load these files with  `np.load(<filename>, allow_pickle=True).item()`
  - *participantsColor.npy* 
    - Dictionary with each key storing the participant's response to color-pairs as a matrix.
  - *participantsEmotion.npy*
    - Dictionary with each key storing the participant's response to emotion-pairs as a matrix.
  - *participantsEmotionColorAssociation.npy*
    - Dictionary with each key storing the participant's emotion to color association as a matrix. 
- load *allColorResponses.npy* with `np.load(<filename>, allow_pickle=True)`
  - Stores the response for each participant as an array.
- Check dataloader module for the further details on how to read the matrix.
---

