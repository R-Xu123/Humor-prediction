# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 23:13:15 2021

@author: xuruizi
"""

import pandas as pd
import numpy as np
import scipy
import sklearn
from collections import Counter
import matplotlib.pyplot as plt
from gensim import corpora
import gensim

import nltk
#nltk.downloader.download('vader_lexicon')
#nltk.downloader.download('wordnet')
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import wordnet as wn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import CoherenceModel
from scipy.sparse import hstack

from readability import Readability

##start

#load and split training data
filepath = './train.csv'
dataframe = pd.read_csv(filepath)
print('training data size:', len(dataframe))
#print(dataframe)
train_ratio=0.7
random_seed=100
train_dataframe = dataframe.sample(frac= train_ratio, random_state=100) 
valid_dataframe = dataframe.drop(train_dataframe.index)
print('training set size:', len(train_dataframe))
print('validation set size:', len(valid_dataframe))

#load test data
test_filepath = './test.csv'
test_dataframe = pd.read_csv(test_filepath)
print('test set size:', len(test_dataframe))
# print(test_dataframe)

#stop words
en_stop = set(nltk.corpus.stopwords.words('english'))









##topic modeling part
#####################################################################################

# take out all title texts in a list
titles = []
for index, row in train_dataframe.iterrows():
#     print (index, row['original'], '|', row['edit'])
    title_text = row['original'].replace('<', '').replace('/>', '')
    titles.append( title_text )
    
# print (titles)



# process a text string into a list of tokens

def prepare_text_for_lda(text):
    # convert all words into lower case, split by white space
    tokens = text.strip().lower().split()
    
    #  remove words with 1 or 2 letters (small words, punctuation)
    tokens = [token for token in tokens if len(token) > 2]
    
    # remove English stopwords (as defined by NLTK)
    tokens = [token for token in tokens if token not in en_stop]
    return tokens

# convert the corpus into a sparse matrix format for gensim

text_data = []
for title in titles:
    title = prepare_text_for_lda(title)
    text_data.append(title)
    
#print(text_data)


dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
#print(corpus)

# train latent Dirichlet topic model
# adjust this number to control how many words, very important
NUM_TOPICS = 7
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)


topics = ldamodel.print_topics()#num_words=8)
for topic in topics:
    print(topic)

 

# predict topic distribution of a new title
new_doc = 'russia white house'
new_doc = prepare_text_for_lda(new_doc)
new_doc_bow = dictionary.doc2bow(new_doc)
predicted_topics = ldamodel.get_document_topics(new_doc_bow)
print('topic distribution:\n', predicted_topics)
print('topic distribution shoulld sum up to 1:', sum([v for k, v in predicted_topics]))

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=ldamodel, texts=text_data, dictionary=dictionary, coherence='u_mass')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

#print("experiment show topc: ")
#x=ldamodel.show_topic(49)
#print(x)

# extract words in topics
topic_words=[]
i=0
while i < NUM_TOPICS:
    for tuples in ldamodel.show_topic(i):
        topic_words.append(tuples[0])
    i+=1
print(topic_words)  
topic_words=set(topic_words)
topic_words=list(topic_words)
#print(topic_words)
punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
'''for word in topic_words:
    for lt in word:
        if lt in punc:
            word=word.replace(lt,'')'''
print('\n the topic words are \n')
#topic_words=(['trump','hillary','russia','gop','republican','republicans','comey'])
print(topic_words)
####################################################################################







##build regression model
#################################################################################

#build feature extractor
# get entire raw text in training corpus, including title and edit words (for learning vocabulary and IDF)

def get_raw_text(df):
    corpus = []
    for index, row in df.iterrows():
        title = row['original'].replace('<', '').replace('/>', '')
        edit = row['edit']
        corpus.append( title + ' ' + edit )
    return corpus

train_corpus = get_raw_text(train_dataframe)
# print (train_corpus)

#
train_corpus = get_raw_text(train_dataframe)
#print (train_corpus)

#try different corpus
# vectorizer = TfidfVectorizer(stop_words = None).fit(train_corpus)
vectorizer = CountVectorizer(stop_words = None).fit(train_corpus)
#vectorizer = CountVectorizer(stop_words=None).fit(topic_words)
#print(vectorizer.vocabulary_)


#extract features from train and validation data

# helper function: separate each title into (original_word, context), where context is the title text without original word 

def separate_original_word_from_title(df):
    original_words = []
    contexts = []
    for index, row in df.iterrows():
        title = row['original']
        start_position = title.find('<')
        end_position = title.find('/>')
        original_words.append(title[start_position+1 : end_position])
        contexts.append(title[:start_position] + title[end_position+2 :])
    return original_words, contexts






#alternative prediction using sentiment and other methods to be combined with topic words
def sentiment_scores(words):
    sentiment=SentimentIntensityAnalyzer()
    sentiment_dict=sentiment.polarity_scores(words)
    #return(sentiment_dict['compound'])
    
    if sentiment_dict['compound'] >= 0.05 : 
        return(1) #positive
  
    elif sentiment_dict['compound'] <= - 0.05 : 
        return(-1) #negative
  
    else : 
        return(0) #neutral
    


# sentiment in original words
def create_alternative_df(df):
    original_sent=[]
    original_read=[]
    #edit_sent=[]
    sentence_len=[]
    original_word_len=[]
    edit_len=[]
    o_words,context= separate_original_word_from_title(df)
 #sentiment
    for x in df['original']:
        original_sent.append(sentiment_scores(x))
        
 #readability, slow things down considerabily
    '''for sent in df['original']:
        r=Readability(sent*50)
        fk=r.flesch_kincaid()
        original_read.append(fk.score)'''

 #lengths
    for y in df['original']:
        sentence_len.append(len(y))
    
    for w in o_words:
        original_word_len.append(len(w))
        
    for z in df['edit']:
        edit_len.append(len(z))
    

    temp_data={'original_sent':original_sent, 'sentence_len':sentence_len,'edit_len':edit_len}
    #temp_data={'original_sent':original_sent, 'original_read':original_read}
    sent_df=pd.DataFrame(temp_data)
    return sent_df


#t_x=create_alternative_df(train_dataframe)
#v_x=create_alternative_df(valid_dataframe)






# construct sparse feature matrix

def construct_feature_matrix(df, vectorizer):
    edit_words = df['edit'].tolist()
    original_words, contexts = separate_original_word_from_title(df)
    #process context here
   # print("context")
   # print(contexts[:20])
   
    # using sentiment and alternative predictions
    Q=create_alternative_df(df)
    
    i=0
    for sent in contexts:
        #print(sent)
        word_list=sent.split()
        result=[word for word in word_list if word.lower() in topic_words]
        contexts[i]=' '.join(result)
        i+=1
   
    print("\n test: processed context \n")
    print(contexts[:20])
    # here the dimensionality of X is len(df) x |V|
    X = vectorizer.transform(contexts)
    #X = vectorizer.transform(topic_words)
    Y = vectorizer.transform(original_words)
    Z = hstack([X,Y])
    
    Z= hstack([Z,Q.to_numpy()])
    
    #Z=(hstack([Z,edit_words]))
#     X = vectorizer.transform(contexts)
    print('Z;!!!!!!!!!!!!!')
    print(Z)
    return Z



#prediction target
train_Y = train_dataframe['meanGrade']
valid_Y = valid_dataframe['meanGrade']

# Construct feature matrices for training and validation data
train_X = construct_feature_matrix(train_dataframe, vectorizer)
print(train_X.shape)
valid_X = construct_feature_matrix(valid_dataframe, vectorizer)
print(valid_X.shape)
test_X = construct_feature_matrix(test_dataframe, vectorizer)
print(test_X.shape)
# print (train_X)




# train model and evaluate
# Construct feature matrices for training and validation data
# train a regression model of choice
# need tune
model = Ridge(alpha=3.5,tol=0.0001).fit(train_X, train_Y)
#model = LinearRegression(normalize=True).fit(train_X, train_Y)
#model = Lasso(alpha=10).fit(train_X, train_Y)
#model = ElasticNet(alpha=100000, l1_ratio=0.9, warm_start=True).fit(train_X,train_Y)
print (model.intercept_)
print (model.coef_.shape)

#print("is it here???????????????????")

# Evaluate model on validation set
valid_Y_hat = model.predict(valid_X)
#valid_Y_hat = model.predict(train_X)
rmse = np.sqrt(sklearn.metrics.mean_squared_error(valid_Y, valid_Y_hat))
print('RMSE on validation set:', rmse)

# Evaluate model on training set: 


train_Y_hat = model.predict(train_X)
rmse = np.sqrt(sklearn.metrics.mean_squared_error(train_Y, train_Y_hat))
print('RMSE on training set:', rmse)



#write to csv
# helper function: write out prediction values into a csv format file
# params:
#     df: dataframe, where each row is a test example, with column 'id' as data id
#     pred: a list or 1-d array of prediction values
#     filepath: the output file path
# return:
#     None

def write_test_prediction(df, pred, filepath):
    with open(filepath, 'w') as outfile:
        outfile.write('{},{}\n'.format('id', 'pred'))
        for index, row in df.iterrows():
            outfile.write('{},{}\n'.format(row['id'], pred[index]))







