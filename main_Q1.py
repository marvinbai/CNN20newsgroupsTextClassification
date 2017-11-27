'''
Author: Xiaoyu Bai
Date: Nov 24, 2017
This program requires the installation of scikit-learn package.
'''

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))


# Fetch the text data.
newsgroups_train = fetch_20newsgroups(subset = 'train', shuffle = True)
newsgroups_test = fetch_20newsgroups(subset = 'test', shuffle = True)

# Feature extraction (text vectorization) using term frequency - inverse document frequency.
tfidf_vect = TfidfVectorizer(stop_words = 'english')
x_train = tfidf_vect.fit_transform(newsgroups_train.data)
x_test = tfidf_vect.transform(newsgroups_test.data)
print('Number of training data is ' + str(x_train.shape[0]))
print('Number of test data is ' + str(x_test.shape[0]))
print('Data dimension is ' + str(x_train.shape[1]))
print()

# Build and fit models.

# 1. Multinomial Naive Bayes.
NB_clf = MultinomialNB(alpha = 0.03).fit(x_train, newsgroups_train.target)
predicted_test = NB_clf.predict(x_test)
predicted_train = NB_clf.predict(x_train)
print('========== 1. Multinomial Naive Bayes ==========')
print('The F-1 score for test query is ' + str(metrics.f1_score(newsgroups_test.target, predicted_test, average = 'macro')))
print('Training accuracy of naive bayes model is ' + str(np.mean(predicted_train == newsgroups_train.target)))
print('Test accuracy of naive bayes model is ' + str(np.mean(predicted_test == newsgroups_test.target)))
print('')
# show_top10(NB_clf, tfidf_vect, newsgroups_train.target_names) # Show 10 most informative features.

# 2. K-Nearest-Neighbors.
knn_clf = KNeighborsClassifier(n_neighbors = 200)
knn_clf.fit(x_train, newsgroups_train.target)
predicted_test = knn_clf.predict(x_test)
predicted_train = knn_clf.predict(x_train)
print('========== 2. K-Nearest-Neighbors ==========')
print('The F-1 score for test query is ' + str(metrics.f1_score(newsgroups_test.target, predicted_test, average = 'macro')))
print('Training accuracy of KNN model is ' + str(np.mean(predicted_train == newsgroups_train.target)))
print('Test accuracy of KNN model is ' + str(np.mean(predicted_test == newsgroups_test.target)))
print('')

# 3. Random Forest.
rf_clf = RandomForestClassifier(n_estimators = 200, max_depth = 100)
rf_clf.fit(x_train, newsgroups_train.target)
predicted_test = rf_clf.predict(x_test)
predicted_train = rf_clf.predict(x_train)
print('========== 3. Random Forest ==========')
print('The F-1 score for test query is ' + str(metrics.f1_score(newsgroups_test.target, predicted_test, average = 'macro')))
print('Training accuracy of random forest model is ' + str(np.mean(predicted_train == newsgroups_train.target)))
print('Test accuracy of random forest model is ' + str(np.mean(predicted_test == newsgroups_test.target)))
print('')

# 4. Linear Support Vector Machine
svm_clf = LinearSVC(loss = 'hinge', penalty = 'l2', tol = 1e-4, max_iter = 1000)
svm_clf.fit(x_train, newsgroups_train.target)
predicted_test = svm_clf.predict(x_test)
predicted_train = svm_clf.predict(x_train)
print('========== 4. Support Vector Machine with Linear Kernel ==========')
print('The F-1 score for test query is ' + str(metrics.f1_score(newsgroups_test.target, predicted_test, average = 'macro')))
print('Training accuracy of SVM model is ' + str(np.mean(predicted_train == newsgroups_train.target)))
print('Test accuracy of SVM model is ' + str(np.mean(predicted_test == newsgroups_test.target)))
print('')
# show_top10(svm_clf, tfidf_vect, newsgroups_train.target_names) # Show 10 most informative features.


