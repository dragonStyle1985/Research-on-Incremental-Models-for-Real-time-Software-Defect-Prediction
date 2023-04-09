import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from util import stats, read, fit

pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 100)

if __name__ == '__main__':
    x, y = read(r'.\jit\input\bugzilla.csv')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
    model = SGDClassifier(penalty="l2", alpha=0.1, max_iter=800)
    model = fit(x_train, y_train, model, np.unique(y))

    # Evaluate the model
    y_pred = model.predict(x_test)
    stats(y_pred, y_test)
