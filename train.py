"""
Muhammet Kara 171805036
Tuğçe Çördük 171805006
Furkan Gümrükçü 171805057
"""

from algorithms import run_algorithms
from matplotlib import pyplot as plt
from data import split_hold_ou, split_cross_val
import numpy as np

def plot(x_test, y_test, predicts, mse, r2, title):
    plt.figure(figsize=(12, 10))
    plt.title(title)
    plt.scatter(range(0, len(x_test.index)), y_test, color="red", label="Real values")
    plt.plot(range(0, len(x_test.index)), predicts, color="blue", label="Predictions")
    plt.figtext(.8, .91, "mse: " + str(round(mse, 4)))
    plt.figtext(.8, .89, "   r2: " + str(round(r2, 2)))
    plt.legend()
    plt.show()

def bar_plot(values, title):
    k = 10
    fold_numbers_str = [str(i + 1) for i in range(k)]
    plt.figure(figsize=(12, 10))
    plt.bar(fold_numbers_str, values)
    plt.title(title)
    plt.figtext(.8, .89, "   avg: " + str(round(np.mean(values), 2)))
    plt.xlabel('Fold Numbers')
    plt.ylabel('R^2 Scores')
    plt.show()

def train_with_cross_val(x, y, test_size=.2, plot_str="", print_stats=True):
    # split data
    x_train, x_test, y_train, y_test = split_hold_ou(x, y, test_size=test_size)
    # run all algorithms
    results = run_algorithms(x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy(), print_stats=print_stats)

    # plot results
    for key in results:
        if key == "Support Vector":
            plot(x_test, results[key]["y_olcekli"], results[key]["predicts"], results[key]["mse"], results[key]["r2"],
                 key + " Regression " + plot_str)
        else:
            plot(x_test, y_test, results[key]["predicts"], results[key]["mse"], results[key]["r2"],
                 key + " Regression " + plot_str)

    return results


def train_with_kfold(x, y, k=3, plot_str="", print_stats=False):
    # split data with k-fold
    kf_indices = split_cross_val(x, k)

    kfold_results = []
    for i, (train_idx, test_idx) in enumerate(kf_indices):
        # take k. fold values
        x_train, x_test, y_train, y_test = x.iloc[train_idx, :], x.iloc[test_idx, :], y.iloc[train_idx, :], y.iloc[test_idx, :]
        # run all algorithms at k. fold data
        results = run_algorithms(x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy(), print_stats=print_stats)
        """
        # plot each algorithm seperately
        for key in results:
            if key == "Support Vector":
                plot(x_test, results[key]["y_olcekli"], results[key]["predicts"], results[key]["mse"], results[key]["r2"], key + " Regression " + plot_str + " k-fold " + str(i+1))
            else:
                plot(x_test, y_test, results[key]["predicts"], results[key]["mse"], results[key]["r2"], key + " Regression " + plot_str + " k-fold " + str(i+1))
        """
        
        kfold_results.append(results)

    avg_mse = 0
    avg_r2 = 0
    
    algorithms_r2_results = np.zeros((8, k))
    # transform results into matrix for barplot
    for i, result in enumerate(kfold_results):
        for j, algorithm in enumerate(result):
            avg_mse += result[algorithm]["mse"]
            avg_r2 += result[algorithm]["r2"]
            
            algorithms_r2_results[j, i] = result[algorithm]["r2"]

    avg_mse = avg_mse / (k*8)
    avg_r2 = avg_r2 / (k*8)

    # plot barplots
    algo_names = ["Lineer", "Ridge", "Lasso", "Support Vector", "Decision Tree", "Random Forest", "Elastic Net", "KNN"]
    for i in range(len(algorithms_r2_results)):
        bar_plot(algorithms_r2_results[i, :], algo_names[i] + " Regression " + plot_str)

    return (algorithms_r2_results, avg_mse, avg_r2)