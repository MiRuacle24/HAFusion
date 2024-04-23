import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_metrics(y_pred, y_test):
    y_pred[y_pred < 0] = 0
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, np.sqrt(mse), r2


def regression(X_train, y_train, X_test, alpha):
    reg = linear_model.Ridge(alpha=alpha)
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    return y_pred


def kf_predict(X, Y):
    kf = KFold(n_splits=25, shuffle=True, random_state=2024)
    y_preds = []
    y_truths = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        y_pred = regression(X_train, y_train, X_test, 1)
        y_preds.append(y_pred)
        y_truths.append(y_test)

    return np.concatenate(y_preds), np.concatenate(y_truths)


def predict_regression(embs, labels, display=False):
    y_pred, y_test = kf_predict(embs, labels)
    mae, rmse, r2 = compute_metrics(y_pred, y_test)
    if display:
        print("MAE:  %.3f" % mae)
        print("RMSE: %.3f" % rmse)
        print("R2:   %.3f" % r2)
    return mae, rmse, r2


def do_tasks(embs, display=True):
    if display:
        print("Check-In Prediction: ")
    check_in_label = np.load("./data_Chi/check_counts.npy")
    region_list = []
    embs_list = []
    chk_label = []
    for i in range(len(embs)):
        if check_in_label[i] > 0:
            region_list.append(i)
            embs_list.append(embs[i])
            chk_label.append(check_in_label[i])
    embs_list = np.array(embs_list)
    chk_label = np.array(chk_label)

    check_mae, check_rmse, check_r2 = predict_regression(embs_list, chk_label, display=display)

    return check_mae, check_rmse, check_r2