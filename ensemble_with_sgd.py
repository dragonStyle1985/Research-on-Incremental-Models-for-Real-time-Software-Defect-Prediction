from sklearn.model_selection import train_test_split

from csce import building_under_sampling_data, ensemble_hybrid_tree
from util import prediction, read, stats

if __name__ == "__main__":
    file_path = r'.\jit\input\bugzilla.csv'
    x_data, y_data = read(file_path)
    print(len(x_data), len(y_data))
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=4)
    tree_list = ensemble_hybrid_tree(50, 1)
    ftree_list = building_under_sampling_data(x_train.values.tolist(), y_train.values.tolist(), 0.5, tree_list)
    y_pred = prediction(ftree_list, x_test, 0.01)

    print(y_pred)
    print(y_test)

    stats(y_pred, y_test)
