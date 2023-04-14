from typing import List

import numpy as np
import pandas as pd
import warnings
from scipy.io import arff
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore', message="X has feature names, but .* was fitted without feature names")


def data_process(arff_data):
    """
    Divide the dataset into attributes and labels.
    :param arff_data:
    :return:
    """
    data = arff.loadarff(arff_data)
    data1 = data[0].tolist()
    attribute_data_list = []
    label_data_list = []
    for each_i in data1:
        temp_list = []
        for each_j in range(len(each_i) - 1):
            temp_list.append(each_i[each_j])
        attribute_data_list.append(temp_list)
        if each_i[-1] == b'Y':
            label_data_list.append(1)
        else:
            label_data_list.append(0)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_min_max = min_max_scaler.fit_transform(attribute_data_list)
    x = x_min_max.tolist()
    return x, label_data_list


# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.x_buffer = []
        self.y_buffer = []

    def add(self, x, y):
        if len(self.x_buffer) >= self.capacity:
            self.x_buffer.pop(0)
            self.y_buffer.pop(0)
        self.x_buffer.append(x)
        self.y_buffer.append(y)

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.x_buffer), size=batch_size)
        x_samples = np.vstack([self.x_buffer[idx] for idx in indices])
        y_samples = np.array([self.y_buffer[idx] for idx in indices])
        return x_samples, y_samples


def stats(y_pred, y_test):
    y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
    # y_true contains the true labels and y_pred contains the predicted labels
    accuracy = accuracy_score(y_test, y_pred_binary)
    print("Accuracy:", accuracy)

    report = classification_report(y_test, y_pred_binary)
    print(report)
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)

    # assume y_true and y_pred are the true and predicted labels respectively
    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred_binary)

    # print precision and recall for each class
    for i in range(len(precision)):
        print(f"Class {i} -- Precision: {precision[i]}, Recall: {recall[i]}")


def read(file):
    df = pd.read_csv(file)
    df['commitdate'] = pd.to_datetime(df['commitdate'])
    df['commitdate'] = (df['commitdate'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')
    x = df.iloc[:, :-1]
    y = df['bug']
    return x, y


def fit(x_train, y_train, model, classes):
    rb = ReplayBuffer(capacity=100)

    # Training loop with memory replay
    epochs = 10
    batch_size = 32
    for epoch in range(epochs):
        for i, (index, row) in enumerate(x_train.iterrows()):
            rb.add(row, y_train.iloc[i])

            if i % batch_size == 0 and len(rb.x_buffer) >= batch_size:
                # Sample a batch from the buffer
                experiences = rb.sample(batch_size)
                result = [exp for exp in experiences]
                batch_x = result[0]
                batch_y = result[1]

                # Train the model on the batch
                model.partial_fit(batch_x, batch_y, classes=classes)
    return model


def prediction(tree_list_: List[DecisionTreeClassifier], test_data, threshold):
    """
    :param tree_list_:
    :param test_data:
    :param threshold: the smaller the threshold, the greater the probability of determining it as a positive example.
    :return:
    """
    print_list = []
    predict_list = []
    for i in tree_list_:
        predict_list.append(i.predict(test_data))
    for j in range(len(test_data)):
        count = 0
        for k in range(len(tree_list_)):
            count = predict_list[k][j] + count
        if count >= len(test_data) * threshold:
            print_list.append(1)
        else:
            print_list.append(0)
    return print_list
