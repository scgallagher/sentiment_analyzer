import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import umap.umap_ as umap
from sklearn.metrics.pairwise import cosine_similarity

class LatentSemanticAnalyzer():

    def __init__(self):

        pd.set_option("display.max_colwidth", 200)
        self.dataset = None

    def get_data(self):

        if self.dataset == None:
            self.dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
        documents = self.dataset.data

        return pd.DataFrame({'document':documents})

    def strip_stop_words(self, df):

        nltk.download('stopwords')
        stop_words = stopwords.words('english')

        # tokenization
        tokenized_doc = df['clean_doc'].apply(lambda x: x.split())
        # remove stop-words
        tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
        # de-tokenization
        detokenized_doc = []
        for i in range(len(df)):
            t = ' '.join(tokenized_doc[i])
            detokenized_doc.append(t)

        df['clean_doc'] = detokenized_doc

        return df

    def clean_data(self, df):

        # remove everything except alphabets`
        df['clean_doc'] = df['document'].str.replace("[^a-zA-Z#]", " ")

        # remove short words
        df['clean_doc'] = df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

        # make all text lowercase
        df['clean_doc'] = df['clean_doc'].apply(lambda x: x.lower())

        return self.strip_stop_words(df)

    def learn(self, df):

        vectorizer = TfidfVectorizer(stop_words='english',
        max_features= 1000, # keep top 1000 terms
        max_df = 0.5,
        smooth_idf=True)

        tfidf_matrix = vectorizer.fit_transform(df['clean_doc'])

        # SVD represent documents and terms in vectors
        svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)

        svd_model.fit(tfidf_matrix)

        return vectorizer, tfidf_matrix, svd_model

    def get_top_terms_for_topic(self, vectorizer, topic_term_matrix):

        top_terms = []
        terms = vectorizer.get_feature_names()
        for i, comp in enumerate(topic_term_matrix):
            terms_comp = zip(terms, comp)
            sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
            top_terms.append(' '.join([term_tuple[0] for term_tuple in sorted_terms]))

        return top_terms

if __name__ == '__main__':

    analyzer = LatentSemanticAnalyzer()
    df = analyzer.get_data()
    df = analyzer.clean_data(df)

    vectorizer, tfidf_matrix, svd_model = analyzer.learn(df)

    document_topic_matrix = svd_model.transform(tfidf_matrix)
    topic_term_matrix = svd_model.components_
    print(document_topic_matrix.shape)
    print(topic_term_matrix.shape)

    test_doc = ['thanks for the space chip sale', 'good people know good windows']

    vectorized_test_doc = vectorizer.transform(test_doc)
    print(vectorized_test_doc.shape)

    top_terms_for_topic = analyzer.get_top_terms_for_topic(vectorizer, topic_term_matrix)
    print(top_terms_for_topic)

    cosine_similarity_matrix = cosine_similarity(vectorized_test_doc, topic_term_matrix)
    print(cosine_similarity_matrix)

    topic_indices = cosine_similarity_matrix.argmax(axis=1)
    topic_predictions = [{'topic_index': topic_index, 'top_terms': top_terms_for_topic[topic_index],
                        'cosine_similarity': cosine_similarity_matrix[i][topic_index]}
                        for i, topic_index in enumerate(topic_indices)]

    print(topic_predictions)
