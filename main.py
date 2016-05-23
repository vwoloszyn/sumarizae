# -*- coding: utf8 -*-



import math
import numpy
import scipy.spatial
import word2vec
import nltk
import codecs

from collections import namedtuple
from operator import attrgetter

    

class Sumarizae:
    """
    LexRank: Graph-based Centrality as Salience in Text Summarization
    Source: http://tangra.si.umich.edu/~radev/lexrank/lexrank.pdf
    """
    threshold = 0.6
    epsilon = 0.1
    _stop_words = frozenset()
    model = word2vec.load('journalistic.bin')
    summary=""



    def __init__(self, sentences, sentences_count):


        matrix = self._create_matrix(sentences, self.threshold)
        scores = self.power_method(matrix, self.epsilon)
        ratings = dict(zip(sentences, scores))
        summaryArray = self._get_best_sentences(sentences, sentences_count, ratings)
        self.summary=""  
        for x in summaryArray:
            self.summary = self.summary + " " + x



    def _get_best_sentences(self, sentences, count, rating, *args, **kwargs):
        ItemsCount = len(sentences)
        SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",))

        rate = rating
        if isinstance(rating, dict):
            assert not args and not kwargs
            rate = lambda s: rating[s]

        infos = (SentenceInfo(s, o, rate(s, *args, **kwargs))
            for o, s in enumerate(sentences))


        # sort sentences by rating in descending order
        infos = sorted(infos, key=attrgetter("rating"), reverse=True)
        # get `count` first best rated sentences
        #if not isinstance(count, ItemsCount):
        #    count = ItemsCount(count)
        #infos = count(infos)
        infos= infos[:count]
        # sort sentences by their order in document
        infos = sorted(infos, key=attrgetter("order"))

        return tuple(i.sentence for i in infos)


    def _create_matrix(self, sentences, threshold):
        """
        Creates matrix of shape |sentences|×|sentences|.
        """
        # create matrix |sentences|×|sentences| filled with zeroes
        sentences_count = len(sentences)
        matrix = numpy.zeros((sentences_count, sentences_count))
        degrees = numpy.zeros((sentences_count, ))


        for row in range(sentences_count):
            for col in range(sentences_count):
                if (row <= col):

                    dist = self._compute_distance(sentences[row], sentences[col])
                    #print("\r{0}".format(i))
                    #print ( str(row) + "-"+str(col)+" = "+str(dist))
                    #print(chr(27) + "[2J")
                    #print (numpy.matrix(matrix))
                    matrix[row, col] = dist
                    matrix[col, row] = dist
                if matrix[row, col] > threshold:
                    matrix[row, col] = 1.0
                    degrees[row] += 1
                else:
                    matrix[row, col] = 0

        for row in range(sentences_count):
            for col in range(sentences_count):
                if degrees[row] == 0:
                    degrees[row] = 1

                matrix[row][col] = matrix[row][col] / degrees[row]

        return matrix

    #@staticmethod


    @staticmethod
    def power_method(matrix, epsilon):
        transposed_matrix = matrix.T
        sentences_count = len(matrix)
        p_vector = numpy.array([1.0 / sentences_count] * sentences_count)
        lambda_val = 1.0

        while lambda_val > epsilon:
            next_p = numpy.dot(transposed_matrix, p_vector)
            lambda_val = numpy.linalg.norm(numpy.subtract(next_p, p_vector))
            p_vector = next_p
        #print (matrix.shape)
        #print (p_vector)
        return p_vector





    def _avg_feature_vector(self, words,num_features):
            #function to average all words vectors in a given paragraph
            featureVec = numpy.zeros((num_features,), dtype="float32")
            nwords = 0
            for word in words:
                if word in self.model.vocab:
                    nwords = nwords+1
                    featureVec = numpy.add(featureVec, self.model[word][:num_features])

            if(nwords>0):
                featureVec = numpy.divide(featureVec, nwords)
            
            #print (featureVec[:10])
            return featureVec

        

    def _compute_distance(self,s1, s2):

        sentence_1_avg_vector = self._avg_feature_vector(s1.split(" "), 100)
        sentence_2_avg_vector = self._avg_feature_vector(s2.split(" "), 100)
        distancia = 1 - scipy.spatial.distance.cosine(sentence_1_avg_vector,sentence_2_avg_vector)


        #print ("sent1="+s1)
        #print ("sent2="+s2)        
        #print ("distancia="+str(distancia))


        return  distancia 



###MAIN


SENTENCES_COUNT = 5

text = codecs.open('critica.txt', 'r',"utf-8").read()
nltk.sent_tokenizer=nltk.data.load('tokenizers/punkt/portuguese.pickle')
sentences = nltk.sent_tokenizer.tokenize(text)



summary = Sumarizae(sentences,SENTENCES_COUNT).summary
print summary
