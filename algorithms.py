
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def get_mse(pred, y):
    return np.mean((pred - y) ** 2)


def lineer_reg(x_train, x_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    predicts = lr.predict(x_test)
    mse = np.mean((predicts - y_test) ** 2)
    r2 = r2_score(y_test, predicts)
    return predicts, mse, r2


# RIDGE REGRESSION
from sklearn.linear_model import Ridge


def ridge_reg(x_train, x_test, y_train, y_test):
    ridgeReg = Ridge(alpha=0.05, normalize=True)
    ridgeReg.fit(x_train, y_train)
    pred = ridgeReg.predict(x_test)
    mse = get_mse(pred, y_test)
    r2 = r2_score(y_test, pred)
    return pred, mse, r2


# LASSO REGRESSION
from sklearn.linear_model import Lasso


def lasso_reg(x_train, x_test, y_train, y_test):
    lassoReg = Lasso(alpha=0.0001, normalize=False)
    lassoReg.fit(x_train, y_train)
    predicts = lassoReg.predict(x_test)
    mse = get_mse(y_test, predicts)
    r2 = r2_score(y_test, predicts)
    return predicts, mse, r2


# SUPPORT VECTOR
# verilerin ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def support_vector(x_train, x_test, y_train, y_test):
    sc1 = StandardScaler()
    x_olcekli = sc1.fit_transform(x_test)
    sc2 = StandardScaler()
    y_olcekli = np.ravel(sc2.fit_transform(y_test.reshape(-1, 1)))
    svr_reg = SVR(kernel='rbf')
    svr_reg.fit(x_olcekli, y_olcekli)
    predicts = svr_reg.predict(x_olcekli)
    mse = get_mse(predicts, y_olcekli)
    r2 = r2_score(y_olcekli, predicts)
    return predicts, mse, r2, y_olcekli


from sklearn.tree import DecisionTreeRegressor


def decision_tree(x_train, x_test, y_train, y_test):
    r_dt = DecisionTreeRegressor(random_state=0)
    r_dt.fit(x_train, y_train)
    predicts = r_dt.predict(x_test)
    mse = np.mean((predicts - y_test) ** 2)
    r2 = r2_score(y_test, predicts)
    return predicts, mse, r2


from sklearn.ensemble import RandomForestRegressor


def random_forest(x_train, x_test, y_train, y_test):
    rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
    rf_reg.fit(x_train, y_train.ravel())
    predicts = rf_reg.predict(x_test)
    mse = np.mean((predicts - y_test) ** 2)
    r2 = r2_score(y_test, predicts)
    return predicts, mse, r2


from sklearn.linear_model import ElasticNet


def elastic_net_reg(x_train, x_test, y_train, y_test):
    elastic_net=ElasticNet(alpha=0.0001,l1_ratio=0.5,random_state=42)
    elastic_net.fit(x_train,y_train)
    predicts = elastic_net.predict(x_test)
    mse = np.mean((predicts - y_test) ** 2)
    r2 = r2_score(y_test, predicts)
    return predicts, mse, r2


from sklearn.neighbors import  KNeighborsRegressor


def knn_reg(x_train, x_test, y_train, y_test):
    reg = KNeighborsRegressor(n_neighbors=3)
    reg.fit(x_train, y_train)
    
    y_pred = reg.predict(x_test)
    mse = np.mean((y_pred - y_test) ** 2)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mse, r2


def run_algorithms(x_train, x_test, y_train, y_test, print_stats=True):
    results = {}

    if print_stats: print("\n----- Lineer Regression -----")
    predicts, mse, r2 = lineer_reg(x_train, x_test, y_train, y_test)
    results["Lineer"] = {"predicts": predicts, "mse": mse, "r2": r2}
    if print_stats: print("mse= ", mse, "\nr2= ",r2)

    if print_stats: print("\n----- Ridge Regression -----")
    predicts, mse, r2 = ridge_reg(x_train, x_test, y_train, y_test)
    results["Ridge"] = {"predicts": predicts, "mse": mse, "r2": r2}
    if print_stats: print("mse= ", mse, "\nr2= ",r2)

    if print_stats: print("\n----- Lasso Regression -----")
    predicts, mse, r2 = lasso_reg(x_train, x_test, y_train, y_test)
    results["Lasso"] = {"predicts": predicts, "mse": mse, "r2": r2}
    if print_stats: print("mse= ", mse, "\nr2= ",r2)

    if print_stats: print("\n----- Support Vector Regression -----")
    predicts, mse, r2, y_olcekli = support_vector(x_train, x_test, y_train, y_test)
    results["Support Vector"] = {"predicts": predicts, "mse": mse, "r2": r2, "y_olcekli": y_olcekli}
    if print_stats: print("mse= ", mse, "\nr2= ",r2)

    if print_stats: print("\n----- Decision Tree Regression -----")
    predicts, mse, r2 = decision_tree(x_train, x_test, y_train, y_test)
    results["Decision Tree"] = {"predicts": predicts, "mse": mse, "r2": r2}
    if print_stats: print("mse= ", mse, "\nr2= ",r2)

    if print_stats: print("\n----- Random Forest Regression -----")
    predicts, mse, r2 = random_forest(x_train, x_test, y_train, y_test)
    results["Random Forest"] = {"predicts": predicts, "mse": mse, "r2": r2}
    if print_stats: print("mse= ", mse, "\nr2= ",r2)
    
    if print_stats: print("\n----- Elastic Net Regression -----")
    predicts, mse, r2 = elastic_net_reg(x_train, x_test, y_train, y_test)
    results["Elastic Net"] = {"predicts": predicts, "mse": mse, "r2": r2}
    if print_stats: print("mse= ", mse, "\nr2= ",r2)
    
    if print_stats: print("\n----- KNN Regression -----")
    predicts, mse, r2 = knn_reg(x_train, x_test, y_train, y_test)
    results["KNN"] = {"predicts": predicts, "mse": mse, "r2": r2}
    if print_stats: print("mse= ", mse, "\nr2= ",r2)

    return results
