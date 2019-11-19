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

        X = vectorizer.fit_transform(df['clean_doc'])
        print(type(df['clean_doc']))

        # SVD represent documents and terms in vectors
        svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)

        svd_model.fit(X)

        return vectorizer, svd_model

if __name__ == '__main__':

    analyzer = LatentSemanticAnalyzer()
    df = analyzer.get_data()
    df = analyzer.clean_data(df)
    vectorizer, svd_model = analyzer.learn(df)

    terms = vectorizer.get_feature_names()

    test_doc = ['thanks for the space chip sale']

    vectorized_test_doc = vectorizer.transform(test_doc)
    print(vectorized_test_doc)

    # for i, comp in enumerate(svd_model.components_):
    #     terms_comp = zip(terms, comp)
    #     sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    #     print("Topic "+str(i)+": ")
    #     for t in sorted_terms:
    #         print(t[0], end=' ')
    #     print('\n')
