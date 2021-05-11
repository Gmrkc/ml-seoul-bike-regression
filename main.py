
from data import load_data
from train import train_with_cross_val, train_with_kfold

           
for i in range(1 ,3, 1):
    if i == 1:
        print("############# Dataset 1: SeoulBikeData #############")
        file = 'SeoulBikeData.csv'
    else:
        input("Press enter to start second dataset training.")
        print("############# Dataset 2: OnlineNewsPopularity #############")
        file = 'OnlineNewsPopularity.csv'
     
    print(">>>> Question 1-2")
    # read data from file and split into train and test groups
    # generate raw and preprocessed dataframes
    x_raw, y_raw = load_data(file, preprocess=False, filter="no_filter")
    x, y = load_data(file, preprocess=True, filter="no_filter")
    print("Question 1-2 done")
    
    print(">>>> Question 3")
    # train with cross validation %80 train %20 test
    # train with raw data
    raw_data_results = train_with_cross_val(x_raw, y_raw, test_size=.2, plot_str=" (Dataset " + str(i) + " Question 3 - Raw Data)")
    # train with preprocessed data
    proccessed_data_results = train_with_cross_val(x, y, test_size=.2, plot_str=" (Dataset " + str(i) + " Question 3 - Preproccessed Data)")
    print("Question 3 done")
    
    print(">>>> Question 4")
    k = 10
    # train with 10-fold
    # train with raw data
    raw_algorithms_r2_results, raw_mse, raw_r2 = train_with_kfold(x_raw, y_raw, k, " (Dataset " + str(i) + " Question 4 - Raw Data)")
    print("raw data kfold average mse: ", raw_mse, "average r2:", raw_r2)
    # train with preprocessed data
    proccessed_algorithms_r2_results, mse, r2 = train_with_kfold(x, y, k, " (Dataset " + str(i) + " Question 4 - Preproccessed Data)")
    print("proccessed data kfold average mse: ", mse, "average r2:", r2)
    print("Question 4 done")
    
    print(">>>> Question 5")
    # dataset half sized with cross correlation
    x_raw, y_raw = load_data(file, preprocess=False, filter="corr")
    x, y = load_data(file, preprocess=True, filter="corr")
    # train with 10-fold
    # train with raw data
    raw_algorithms_r2_results, raw_mse, raw_r2 = train_with_kfold(x_raw, y_raw, k, " (Dataset " + str(i) + " Question 5 - Raw Data half w. corr)")
    print("raw data kfold average mse: ", raw_mse, "average r2:", raw_r2)
    # train with preprocessed data
    proccessed_algorithms_r2_results, mse, r2 = train_with_kfold(x, y, k, " (Dataset " + str(i) + " Question 5 - Preproccessed Data, half w. corr)")
    print("proccessed data kfold average mse: ", mse, "average r2:", r2)
    print("Question 5 done")
    
    print(">>>> Question 6")
    #dataset half sized with mse
    x_raw, y_raw = load_data(file, preprocess=False, filter="mse")
    x, y = load_data(file, preprocess=True, filter="mse")
    # train with 10-fold
    # train with raw data
    raw_algorithms_r2_results, raw_mse, raw_r2 = train_with_kfold(x_raw, y_raw, k, " (Dataset " + str(i) + " Question 6 - Raw Data half w. mse)")
    print("raw data kfold average mse: ", raw_mse, "average r2:", raw_r2)
    # train with preprocessed data
    proccessed_algorithms_r2_results, mse, r2 = train_with_kfold(x, y, k, " (Dataset " + str(i) + " Question 6 - Preproccessed Data, half w. mse)")
    print("proccessed data kfold average mse: ", mse, "average r2:", r2)
    print("Question 6 done")
