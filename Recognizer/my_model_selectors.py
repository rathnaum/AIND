import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word 
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)
        
class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """       
    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        """Initialize the variables to hold values"""
        candidateModel = None
        candidateBIC = float('inf')
        
        """Loop for optimal BIC value and model"""
        for n in range(self.min_n_components, self.max_n_components+1):
            logL = None
            model = self.base_model(n)
            if model is None:
                continue
            try:  # hmmlearn stability issues
                logL = model.score(self.X, self.lengths)
            except:
                continue
            # calculate BIC
            BIC = -2 * logL + (n**2 + (2 * len(self.X[0]) * n)) * np.log(len(self.X))
            if  BIC < candidateBIC:
                candidateBIC = BIC
                candidateModel = model
        return candidateModel

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        candidateDIC = float('-inf')
        candidateModel = None

        for n in range(self.min_n_components, self.max_n_components+1):
            M = 1.0
            try:
                model = self.base_model(n)
                LogL0 = model.score(self.X, self.lengths)
            except:
                continue

            logsum = 0
            for word in self.hwords.keys():
                if word != self.this_word:
                    x2, l2 = self.hwords[word]
                    try:
                        wordlsum = model.score(x2, l2)
                        M = M + 1.0
                    except:
                        wordlsum = 0
                    logsum += wordlsum
            # calculate DIC
            newDIC = LogL0 - (1/(M-1)) * logsum * 1.0

            if newDIC > candidateDIC:
                candidateDIC = newDIC
                candidateModel = model
        return candidateModel



class SelectorCV(ModelSelector):
    ''' Select best model based on average log Likelihood of cross-validation folds'''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Model selection using CV
        split_method = KFold(n_splits = min(len(self.lengths),3))
        CandidateLL_avg = -float('Inf')
        CandidateModel = None

        """loop with range of states, find avg logl. divide and run test, train on data sets."""
        newModel = None
        for n in range(self.min_n_components, self.max_n_components + 1):
            LLSum = 0
            LLcount = 1
            matchModel = None
            for CVtrain, CVtest in split_method.split(self.sequences):

                Xtrain, Xtest = [],[]
                for test in CVtest:
                    Xtest += self.sequences[test]
                for train in CVtrain:
                    Xtrain += self.sequences[train]

                Xtrain, Xtest = np.array(Xtrain), np.array(Xtest)
                train_length, test_length = np.array(self.lengths)[CVtrain], np.array(self.lengths)[CVtest]

                try:
                    matchModel = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                               random_state=self.random_state, verbose=False).fit(Xtrain, train_length)
                    LogL = matchModel.score(Xtest, test_length)
                    LLcount += 1
                except:
                    LogL = 0
                LLSum += LogL

            LLAvgLocal = LLSum/(LLcount*1.0)

            if LLAvgLocal>CandidateLL_avg:
                CandidateLL_avg = LLAvgLocal
                CandidateModel = matchModel

        return CandidateModel