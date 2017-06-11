import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # implement the recognizer, return probabilities, guesses
    # for each word from test set, extract (X, lengths) tuple

    # add to dictionary based on probability 
    for word,_ in test_set.get_all_Xlengths().items():
        dictionary = dict()
        X_test, len_test = test_set.get_item_Xlengths(word)
        for wordModel, model in models.items():
            try:
                LogL = model.score(X_test, len_test)
            except:
                LogL = -float('inf')
            dictionary[wordModel] = LogL
        #add our probabilities
        probabilities.append(dictionary) 
        #find best prob
        guesses.append(max(dictionary, key=dictionary.get)) 

    return probabilities, guesses
