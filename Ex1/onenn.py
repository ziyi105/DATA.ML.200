from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import timeit
from sklearn.neighbors import KNeighborsClassifier

# Fetch dataset
def size_mb(docs):
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6

data_train = fetch_20newsgroups(subset='train')
data_test = fetch_20newsgroups(subset='test')

# Vectorize dataset
vectorizer = CountVectorizer()
vectorizer = CountVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)
y_train, y_test = data_train.target, data_test.target

# Define simple baseline classifier
def classify_all_test_data_with_simple_baseline():
    simple_baseline = DummyClassifier(strategy="most_frequent")
    simple_baseline.fit(X_train, y_train)
    prediction = simple_baseline.predict(X_test)
    return prediction

# Compute simple baseline classifier
start = timeit.default_timer()
simple_baseline_prediction = classify_all_test_data_with_simple_baseline()

print("Computational time for the baseline : ", 
            timeit.default_timer() - start)

print("Baseline classification accuracy : ", accuracy_score(y_test, simple_baseline_prediction))

# Implement Simple Nearest Neighbour Classifier
def calculate_euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def nearest_neighbor_classifier(train_data, train_labels, test_sample):
    min_distance = float('inf')
    nearest_neighbor_label = None

    for i, train_sample in enumerate(train_data):
        distance = calculate_euclidean_distance(train_sample, test_sample)

        if distance < min_distance:
            min_distance = distance
            nearest_neighbor_label = train_labels[i]
        
    return nearest_neighbor_label

def predict_nearest_neighbors(train_data, train_labels, test_data):
    predictions = []

    for test_sample in test_data:
        prediction = nearest_neighbor_classifier(train_data, train_labels, test_sample)
        predictions.append(prediction)

    return np.array(predictions)

def nn_prediction(X_train_array, y_train, X_test_array):
    nn_predictions = predict_nearest_neighbors(X_train_array, y_train, X_test_array)
    return nn_predictions

# Take a subset of the dataset
num_samples_to_estimate = 100
X_train_subset = X_train.toarray()[:num_samples_to_estimate]
X_test_subset = X_test.toarray()[:num_samples_to_estimate]

y_train_subset = y_train[:num_samples_to_estimate]
y_test_subset = y_test[:num_samples_to_estimate]

# Predict using self-implemented NN
start = timeit.default_timer()
nn_predictions = nn_prediction(X_train_subset, y_train_subset, X_test_subset)

estimated_total_computational_time = (timeit.default_timer() - start) * len(y_test) / num_samples_to_estimate

print("Number of test samples used : ", num_samples_to_estimate)
print("Estimated Computational time of self-implemented NN : ", 
            estimated_total_computational_time)
print("Classification accuracy of self-implemented NN : ", accuracy_score(y_test_subset, nn_predictions))

# Replace with SKlearn NN Classifier
start = timeit.default_timer()

knn_classifier = KNeighborsClassifier(n_neighbors=1)
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)

print("Computational time of SKlearn NN : ", 
            timeit.default_timer() - start)
print("Classification accuracy of SKlearn NN : ", accuracy_score(y_test, knn_predictions))
