import pandas as pd
import os
import gensim
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import datetime
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import  cosine_similarity
#import kmedoids
import pickle
import nltk
#nltk.download()
#base_dir = "C:/Users/sasthan1/Documents/Suman-SA work/Python code/Resource/"
base_dir= "/data/"
def textPreprocessing(contentmatrix):
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    stop_words.append("Unclear")
    stop_words.append("unclear")
    stop_words.append('.')
    stop_words.append('\'')
    stop_words.append('!')
    stop_words.append('?')
    cleaned_matrix = pd.DataFrame()
    contentmatrix['Sentence'] = contentmatrix['Sentence'].apply(lambda x: x if (not (isinstance(x, datetime.date) or isinstance(x, datetime.time) or isinstance(x, float)  or isinstance(
        x, int))) else "")
    cleaned_matrix['Sentence'] = contentmatrix['Sentence'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, 'v') for word in x.split() if((word not in (stop_words)) and word.isalpha() and len(word) > 2)]))
    cleaned_matrix['Sentence'] = cleaned_matrix['Sentence'].apply(lambda x: ' '.join(word.lower().replace('.', '').replace('\'', '').replace('?', '') for word in x.split()))
    return cleaned_matrix


def main():
    #somecode
    print("\n Initializing word2vec. Download it from here: https://github.com/mmihaltz/word2vec-GoogleNews-vectors")
    global google_word2vec
    google_word2vec = gensim.models.KeyedVectors.load_word2vec_format(base_dir+"GoogleNews-vectors-negative300.bin", binary=True)
    content = pd.read_excel(base_dir+"IDMB_Sentence_score.xlsx", index_col=None)
    preprocessedContent = textPreprocessing(content)
    preprocessedContent = preprocessedContent[:10000]
    preprocessedContent.reset_index(drop=True, inplace=True)
    dist_matrix = np.zeros(shape=(len(preprocessedContent),len(preprocessedContent)))
    google_word2vec.init_sims(replace=True)
    #print(preprocessedContent["Sentence"])
    for i in range(len(preprocessedContent['Sentence'])):
        for j in range(i,len(preprocessedContent['Sentence'])):
	    print i, j
            if(i==j):
                dist_matrix[i][j] = 0
            else:
                dist_matrix[i][j]=dist_matrix[j][i]=  google_word2vec.wmdistance(preprocessedContent['Sentence'][i], preprocessedContent['Sentence'][j])

    #print(dist_matrix)
    path = base_dir + str("distmatrix_IMDB.pkl")
    with open(path, 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump([dist_matrix], f)
    df = pd.DataFrame(dist_matrix)
    df.to_csv("/data/IMDB.csv", ",")


if __name__== "__main__":
    main()
