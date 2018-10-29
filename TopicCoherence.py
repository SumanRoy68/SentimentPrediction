import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import datetime
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import  cosine_similarity
#import kmedoids
import pickle
import nltk

base_dir= "C:/Users/m37/Topic Modeling/"
def main():
    print("\n Initializing word2vec. Download it from here: https://github.com/mmihaltz/word2vec-GoogleNews-vectors")
    global google_word2vec
    google_word2vec = gensim.models.KeyedVectors.load_word2vec_format(base_dir+"GoogleNews-vectors-negative300.bin", binary=True)
    content = pd.read_excel(base_dir+"IDMB_Sentence_score.xlsx", index_col=None)
    preprocessedContent = textPreprocessing(content)
    preprocessedContent = preprocessedContent[:10000]
    #df = pd.read_excel("train_sentiment.xlsx", index_col=None)
    documents = preprocessedContent["Sentence"]
    count_vec = CountVectorizer()
    count_matrix = count_vec.fit_transform(documents)
    #print(count_matrix.shape)
    #D = np.sum(count_matrix)
    #print(D)
    model_nmf= NMF_Model(count_matrix)
    #model_lda = LDA_Model(count_matrix)
    tc= TopicCoherence(model_nmf,10,count_matrix)
    print(tc)
    print(np.mean(tc))

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

def NMF_Model(tf):
    no_topics = 9
    print("Learning NMF Model")
    nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tf)
    return nmf

def LDA_Model(tf):
    no_topics = 9
    lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,
                                    random_state=0).fit(tf)
    return lda

def scoreUCI(i,j,count_matrix):
    a = count_matrix[:,i]
    b= count_matrix[:,j]
    c = a.toarray()*b.toarray()
    d = np.count_nonzero(c)
    e = np.count_nonzero(a.toarray())
    f=  np.count_nonzero(b.toarray())
    w= d*1.0*a.shape[0]/(e*f)
    if(w==0):
        return 0
    else:
        return np.max((np.log(w)),0)
    


def TopicCoherence(model, no_top_words,count_matrix):
    no_topics = 9
    top_words_matrix= np.zeros(shape= (no_topics,no_top_words),dtype=int)
    for topic_idx, topic in enumerate(model.components_):
     #   print("Topic %d:" % (topic_idx))
        top_words_matrix[topic_idx]= [i for i in topic.argsort()[:-no_top_words - 1:-1]]
     #for topic_idx in range(9):
     #   print("Topic %d:" % (topic_idx))
      #   top_words_matrix[topic_idx]=word_frequencies_by_cluster.loc[topic_idx, :].sort_values(ascending=False)[:10]
    topic_coherence = np.zeros(shape=(no_topics,1))
    for i in range(no_topics):
        for j in range(no_top_words):
            for k in range (j,no_top_words):
                print("Calculating topic coherence for" + str(i))
                topic_coherence[i] += scoreUCI(top_words_matrix[i][j],top_words_matrix[i][k],count_matrix)
    topic_coherence *=  2/(no_top_words*(no_top_words-1))
    return topic_coherence


if __name__ == "__main__":
    main()