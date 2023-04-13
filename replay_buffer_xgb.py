import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from util import stats, fit, read
from xgboost import XGBRegressor

pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 100)

if __name__ == '__main__':
    x, y = read(r'.\jit\input\bugzilla.csv')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
    model = XGBRegressor(learning_rate=0.02, n_estimators=600, objective='binary:logistic', silent=True, nthread=1)
    model = fit(x_train, y_train, model, np.unique(y))

    # Evaluate the model
    y_pred = model.predict(x_test)
    stats(y_pred, y_test)
