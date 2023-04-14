import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from csce import ensemble_tree, building_under_sampling_data
from util import prediction, read, stats

if __name__ == '__main__':
    file_path = r'.\jit\input\bugzilla.csv'
    x_data, y_data = read(file_path)
    print(len(x_data), len(y_data))
    split_n = 5
    x_arrays = np.array_split(x_data.values, split_n)
    y_arrays = np.array_split(y_data.values, split_n)
    x_dfs = [pd.DataFrame(array, columns=x_data.columns) for array in x_arrays]
    y_dfs = [pd.DataFrame(array, columns=y_data.to_frame().columns) for array in y_arrays]
    tree_list = []
    x_test_list = []
    y_test_list = []
    for i in range(split_n):
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=4)
        x_test_list.append(x_test)
        y_test_list.append(y_test)
        current_ensemble_tree = ensemble_tree(10)
        current_ensemble_tree = building_under_sampling_data(x_train.values.tolist(), y_train.values.tolist(), current_ensemble_tree)
        tree_list.append(current_ensemble_tree)
        merged_x_test_df = pd.concat(x_test_list, axis=0, ignore_index=True)
        y_pred = prediction(sum(tree_list, []), merged_x_test_df, 0.01)
        stats(y_pred, pd.concat(y_test_list, ignore_index=True))
