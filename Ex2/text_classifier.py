from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Fetch dataset
data_train = fetch_20newsgroups(subset='train')
data_test = fetch_20newsgroups(subset='test')
vocab_sizes = [10, 100, 1000, 5000, 10000]

def evaluate_classifier(vocab_sizes, stop_words=None, use_tfidf=False):
    accuracies = []
    
    for size in vocab_sizes:
        vectorizer = CountVectorizer(max_features=size, stop_words=stop_words)

        X_train = vectorizer.fit_transform(data_train.data)
        X_test = vectorizer.transform(data_test.data)
        y_train, y_test = data_train.target, data_test.target

        if use_tfidf:
            tfidf_transformer = TfidfTransformer()
            X_train_tfidf = tfidf_transformer.fit_transform(X_train)
            X_test_tfidf = tfidf_transformer.transform(X_test)
            X_train, X_test = X_train_tfidf.toarray(), X_test_tfidf.toarray()

        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train )
        predictions = knn.predict(X_test)

        accuracy = accuracy_score(data_test.target, predictions)
        accuracies.append(accuracy)
    return accuracies


full_vocab_accuracy = evaluate_classifier([None], stop_words=None)
full_vocab_accuracy_sw = evaluate_classifier([None], stop_words='english')

limited_vocab_accuracy = evaluate_classifier(vocab_sizes, stop_words=None)
limited_vocab_accuracy_sw = evaluate_classifier(vocab_sizes, stop_words='english')

tfidf_accuracy = evaluate_classifier(vocab_sizes, stop_words=None, use_tfidf=True)

plt.figure(figsize=(12, 6))
plt.plot(vocab_sizes, full_vocab_accuracy * len(vocab_sizes), marker='o', label='Full Vocabulary', color='blue')
plt.plot(vocab_sizes, full_vocab_accuracy_sw * len(vocab_sizes), marker='x', label='Full Vocabulary (Stop Words)', color='green')
plt.title('Full Vocabulary Accuracy vs. Vocabulary Size')
plt.xlabel('Vocabulary Size')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.xticks(vocab_sizes, [str(size) for size in vocab_sizes])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

plt.figure(figsize=(12, 6))
plt.plot(vocab_sizes, limited_vocab_accuracy, marker='o', linestyle='-', label='Limited Vocabulary', color='red')
plt.plot(vocab_sizes, limited_vocab_accuracy_sw, marker='x', linestyle='-', label='Limited Vocabulary (Stop Words)', color='orange')
plt.title('Accuracy vs. Vocabulary Size')
plt.xlabel('Vocabulary Size')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.xticks(vocab_sizes, [str(size) for size in vocab_sizes])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

plt.figure(figsize=(12, 6))
plt.plot(vocab_sizes, tfidf_accuracy, marker='o', label='TF-IDF Weighting', color='purple')
plt.title('Accuracy vs. Vocabulary Size with TF-IDF Weighting')
plt.xlabel('Vocabulary Size')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.xticks(vocab_sizes, [str(size) for size in vocab_sizes])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()