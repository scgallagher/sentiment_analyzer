import random
from collections import Counter

class TopicModeller():

    def __init__(self, num_topics):

        self.num_topics = num_topics

    def probability_topic_given_document(self, topic, document, alpha=0.1):
        return ((self.document_topic_counts[document][topic] + alpha) /
                (self.document_lengths[document] + self.num_topics * alpha))

    def probability_word_given_topic(self, word, topic, beta=0.1):
        return ((self.topic_word_counts[topic][word] + beta) /
               (self.topic_counts[topic] + self.W * beta))

    def sample_from(self, weights):
        # returns i with probability weights[i] / sum(weights)
        total = sum(weights)
        rnd = total * random.random()
        for i, weight in enumerate(weights):
            rnd -= weight
            if rnd <=0:
                return i

    def topic_weight(self, document, word, topic):
        return self.probability_word_given_topic(word, topic) * self.probability_topic_given_document(topic, document)

    def choose_new_topic(self, document, word):
        return self.sample_from([self.topic_weight(document, word, topic) for topic in range(self.num_topics)])

    def learn_model(self, documents):

        # Number of times each topic is assigned to each document
        self.document_topic_counts = [Counter() for _ in documents]

        # Number of times each word is assigned to each topic
        self.topic_word_counts = [Counter() for _ in range(self.num_topics)]

        # Total number of words assigned to each topic
        self.topic_counts = [0 for _ in range(self.num_topics)]

        # Total number of words in each document
        self.document_lengths = [len(d) for d in documents]

        # Number of distinct words
        self.distinct_words = set(word for document in documents for word in document)
        self.W = len(self.distinct_words)

        # Number of documents
        self.D = len(documents)

        self.document_topics = [[random.randrange(self.num_topics) for word in document] for document in documents]

        for d in range(self.D):
            for word, topic in zip(documents[d], self.document_topics[d]):
                self.document_topic_counts[d][topic] += 1
                self.topic_word_counts[topic][word] += 1
                self.topic_counts[topic] += 1

        for iter in range(1000):
            for d in range(self.D):
                for i, (word, topic) in enumerate(zip(documents[d], self.document_topics[d])):
                    # Remove current word/topic from the counts so they don't influence weights
                    self.document_topic_counts[d][topic] -= 1
                    self.topic_word_counts[topic][word] -= 1
                    self.topic_counts[topic] -= 1
                    self.document_lengths[d] -= 1

                    # Choose a new topic based on the weights
                    new_topic = self.choose_new_topic(d, word)
                    self.document_topics[d][i] = new_topic

                    # Add word/topic back to the counts
                    self.document_topic_counts[d][new_topic] += 1
                    self.topic_word_counts[new_topic][word] += 1
                    self.topic_counts[new_topic] += 1
                    self.document_lengths[d] += 1

    def print_top_five(self):

        for k, word_counts in enumerate(self.topic_word_counts):
            for word, count in word_counts.most_common(5):
                if count > 0:
                    print(k, word, count)
                    
    def print_document_topics(self, documents):

        topic_names = ['Big Data and Programming Languages', 'Python and Statistics', 'Databases', 'Machine Learning']

        for document, topic_counts in zip(documents, self.document_topic_counts):
            print(document)
            for topic, count in topic_counts.most_common(5):
                if count > 0:
                    print('\t', topic_names[topic], count)
            print()

if __name__ == '__main__':

    documents = [
        ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
        ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
        ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
        ["R", "Python", "statistics", "regression", "probability"],
        ["machine learning", "regression", "decision trees", "libsvm"],
        ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
        ["statistics", "probability", "mathematics", "theory"],
        ["machine learning", "scikit-learn", "Mahout", "neural networks"],
        ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
        ["Hadoop", "Java", "MapReduce", "Big Data"],
        ["statistics", "R", "statsmodels"],
        ["C++", "deep learning", "artificial intelligence", "probability"],
        ["pandas", "R", "Python"],
        ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
        ["libsvm", "regression", "support vector machines"]
    ]

    modeller = TopicModeller(4)
    modeller.learn_model(documents)
    modeller.print_top_five()
