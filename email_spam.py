from collections import Counter
import os
from nltk.corpus import stopwords
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

data = {}


def make_dictionary(train_dir):
    all_words = []
    emails = [os.path.join(train_dir, file) for file in os.listdir(train_dir)]

    global data

    for file in emails:
        with open(file) as f:
            for line in f:
                words = line.split()
                all_words += words


    frequent = Counter(all_words)

    # list_to_remove = list(frequent)
    list_to_remove = list(frequent)
    stop_words = set(stopwords.words("english"))

    #### remove stop-words and single letters ####

    for item in list_to_remove:
        if item.isalpha() == False or item in stop_words:
            del frequent[item]

    frequent = frequent.most_common(2500)

    count = 0
    for word in frequent:
        data[word[0]] = count
        count += 1


def feature_extraction(train_dir):
    files = [os.path.join(train_dir, file) for file in os.listdir(train_dir)]
    features_matrix = np.zeros((len(files), 2500))
    labels = np.zeros(len(files))
    file_count = 0

    for file in files:
        with open(file) as file_obj:
            for index, line in enumerate(file_obj):
                if index == 2:
                    line = line.split()
                    for word in line:
                        if word in data:
                            features_matrix[file_count, data[word]] = line.count(word)

        labels[file_count] = 0

        #### file names containing 'spmsg' are labelled as 1 ####

        if 'spmsg' in file:
            labels[file_count] = 1
        file_count += 1
    return features_matrix, labels


if __name__ == '__main__':
    training_data = 'train-mails'
    testing_data = 'test-mails'

    make_dictionary(training_data)

    model1 = KNeighborsClassifier(n_neighbors=40)
    model2 = LinearSVC()
    model3 = MultinomialNB()
    model4 = svm.SVC(kernel='linear', C=1.0)
    model5 = LogisticRegression(max_iter=100)
    model6 = tree.DecisionTreeClassifier()
    model7 = RandomForestClassifier()

    training_feature, training_labels = feature_extraction(training_data)
    testing_features, testing_labels = feature_extraction(testing_data)

    model1.fit(training_feature, training_labels)
    model2.fit(training_feature, training_labels)
    model3.fit(training_feature, training_labels)
    model4.fit(training_feature, training_labels)
    model5.fit(training_feature, training_labels)
    model6.fit(training_feature, training_labels)
    model7.fit(training_feature, training_labels)

    ###### Predicting the labels for the test files ######

    result1 = model1.predict(testing_features)
    result2 = model2.predict(testing_features)
    result3 = model3.predict(testing_features)
    result4 = model4.predict(testing_features)
    result5 = model5.predict(testing_features)
    result6 = model6.predict(testing_features)
    result7 = model7.predict(testing_features)

    print('Accuracy of kNN classifier:', accuracy_score(testing_labels, result1) * 100)
    print(confusion_matrix(testing_labels, result1))

    print('Accuracy of LinearSVC classifier:', accuracy_score(testing_labels, result2) * 100)
    print(confusion_matrix(testing_labels, result2))

    print('Accuracy of MultinomialNB classifier:', accuracy_score(testing_labels, result3) * 100)
    print(confusion_matrix(testing_labels, result3))

    print('Accuracy of SVM(linear kernel) classifier:', accuracy_score(testing_labels, result4) * 100)
    print(confusion_matrix(testing_labels, result4))

    print('Accuracy of Logistic regression:', accuracy_score(testing_labels, result5) * 100)
    print(confusion_matrix(testing_labels, result5))

    print('Accuracy of Decision tree classifier:', accuracy_score(testing_labels, result6) * 100)
    print(confusion_matrix(testing_labels, result6))

    print('Accuracy of Random forest classifier:', accuracy_score(testing_labels, result7) * 100)
    print(confusion_matrix(testing_labels, result7))


