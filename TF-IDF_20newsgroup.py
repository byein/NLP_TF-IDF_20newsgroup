#!/usr/bin/python
#-*- coding: utf-8 -*-
"""NLP Homework #3: Twenty Newsgroups Classification

This is a skeleton file for NLP homework.
Please complete your task with respect to the given guideline.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import rand
from sklearn import (datasets, feature_extraction, linear_model, metrics,svm)
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
def train(dataset, target):
    '''
    Train my model with the given training dataset

    Parameters
    ----------
    dataset: list or numpy.array
        The given training dataset

    target: list or numpy.array
        The given training target (true labels)

    Returns
    -------
    object or tuple of objects
        My trained model such as a feature extractor and classifier

    Notes
    -----
    * My name: 김예빈
    * My student ID: 19101198
    * My accuracy (max. 1): 0.699
    * Brief description
      - ml 수업 때 배운 다양한 Classifier를 적용해보았습니다.
      - 약 10개 정도를 검사했고 그 결과 중 0.5 이상의 정확도를 가지는 것들을 선택하였습니다.
      - 추려진 결과 각각에 대해 vectorizer를 TfidfVectorizer로 변경하여 정확도를 높이고자 했고 그 중 가장 높은 정확도를 내는 값을 선택하였습니다.
      - 결과적으로 변경된 부분은 다음과 같습니다.
      - classifier = svm.LinearSVC()
      - vectorizer = feature_extraction.text.TfidfVectorizer()
    * Discussion
      - 과제를 진행하면서 가능하면 0.7 이상의 정확도를 내고자 노력했으나 다양한 방법을 시도해도 0.699까지가 한계였습니다.
      - 아쉬운 마음이 드는데 어떻게 하면 0.7 이상의 정확도를 가질 수 있는지가 너무 궁금합니다.
      - classifier를 ComplementNB()로 변경하여 0.712 획득.
    * Collaborators: 혼자 진행하였습니다.
    * References
      - ml 강의자료와 vectorization_bow 강의자료를 참고했습니다.
      - 그동안 colab에서 실행하여 시스템에서 .py 파일을 돌리기 위해 아래 웹사이트를 참고하여 시스템 환경설정을 실행하였습니다.
        - https://blog.naver.com/PostView.naver?blogId=racoonpapa&logNo=222435398541&redirect=Dlog&widgetTypeCall=true&directAccess=false
      - 사이킷런 웹사이트에서 20 newgroups dataset을 활용하는 부분을 참고했습니다.
        - https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
      - classifier의 정확도를 높이기 위해 사이킷런 웹사이트에서 LinearSVC 를 참고했습니다.
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

    '''

    # PLEASE WRITE YOUR CODE HERE
    # (The following lines are my example, you can remove them.)
    vectorizer = feature_extraction.text.TfidfVectorizer(stop_words='english', ngram_range=(1,3), max_df=1.0, min_df=1, sublinear_tf=True, analyzer='word')
    vector = vectorizer.fit_transform(dataset)
    # classifier = svm.LinearSVC(multi_class='crammer_singer', random_state=0, tol=0.001, C=1.0, fit_intercept=True, intercept_scaling=1.0, class_weight=None, verbose=0)
    classifier = ComplementNB(alpha=0.4)
    classifier.fit(vector, target)
    return (vectorizer, classifier)

def predict(model, dataset):
    '''
    Predict lables of the given test dataset

    Parameters
    ----------
    model: object or tuple of objects
        My trained model such as a feature extractor and classifier

    dataset: list or numpy.array
        The given test dataset

    Returns
    -------
    list or numpy.array
        Predicted labels
    '''

    # PLEASE MODIFY THE FOLLOW IF NECESSARY
    vectorizer, classifier = model
    vector = vectorizer.transform(dataset)
    pred = classifier.predict(vector)
    return pred



if __name__ == '__main__':
    # Load a dataset (Note: For the first time, it spent long time to download the datasets.)
    remove = ('headers', 'footers', 'quotes')
    news20_train = datasets.fetch_20newsgroups(subset='train', remove=remove)
    news20_test  = datasets.fetch_20newsgroups(subset='test',  remove=remove)

    # Train a model and evaluate it
    model = train(news20_train.data, news20_train.target)
    pred = predict(model, news20_test.data)
    accuracy = metrics.balanced_accuracy_score(news20_test.target, pred)

    # Print the results
    print('### My results')
    print(f'* My accuracy: {accuracy:.3}')