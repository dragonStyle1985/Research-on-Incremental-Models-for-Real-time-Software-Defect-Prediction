from sklearn.model_selection import train_test_split

from csce import ensemble_tree, building_under_sampling_data
from util import prediction, read, stats

if __name__ == "__main__":
    file_path = r'.\jit\input\bugzilla.csv'
    x_data, y_data = read(file_path)
    print(len(x_data), len(y_data))
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=4)
    tree_list = ensemble_tree(50)
    ftree_list = building_under_sampling_data(x_train.values.tolist(), y_train.values.tolist(), 0.5, tree_list)
    y_pred = prediction(ftree_list, x_test, 0.01)

    print(y_pred)
    print(y_test)
    # make_result = MakeResult(y_pred, y_test.tolist())
    # print('RE算法在{}数据集上的Prediction、PD、PF、AUC、F1分别为：'.format(file_path))
    # print(make_result.calculatePrediction(), make_result.calculatePD(), make_result.calculatePF(), make_result.calculateF1(), make_result.calculateAUC())
    stats(y_pred, y_test)
