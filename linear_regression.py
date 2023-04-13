from sklearn import linear_model
from sklearn.model_selection import train_test_split

from util import stats, data_process

if __name__ == '__main__':
    x, y = data_process(r'.\NASA\D\PC5.arff')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
    lr_sk_model = linear_model.LogisticRegression(penalty="l2", C=0.1, max_iter=800)
    lr_sk_model.fit(x_train, y_train)
    y_pred = lr_sk_model.predict(x_test)
    stats(y_pred, y_test)
