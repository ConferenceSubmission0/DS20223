import pandas as pd

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score

from sklearn.pipeline import Pipeline

def classifier_DecisionTree(X_train, X_test, y_train):
    pipe = Pipeline([('tree', DecisionTreeClassifier())], verbose=3)
    clf = pipe.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return clf, y_pred

def regressor_DecisionTree(X_train, X_test, y_train):
    pipe = Pipeline([('tree', DecisionTreeRegressor())], verbose=3)
    clf = pipe.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return clf, y_pred

def evaluate_models(y_test, y_pred, classf=0):

    if classf == 0:
            return mean_squared_error(y_test, y_pred)

    else:
            return f1_score(y_test, y_pred, average="weighted")

def models_DT(X, y, classf=0):
    # Split the data into training and test sets
    X_train1, X_test, y_train1, y_test = train_test_split(X, y, test_size=0.20)
    X_train2 = X_train1[:int(len(X_train1)*0.15)]
    y_train2 = y_train1[:int(len(y_train1)*0.15)]

    X_train3_labelled = X_train2.copy()
    y_train3_labelled = y_train2.copy()

    X_train3_unlabelled = X_train1[int(len(X_train1)*0.15):]

    if classf == 0:
        # Train and predict using DecisionTreeRegressor for Model 1
        clf1, y_pred1 = regressor_DecisionTree(X_train1, X_test, y_train1)

        # Train and predict using DecisionTreeRegressor for Model 2
        clf2, y_pred2 = regressor_DecisionTree(X_train2, X_test, y_train2)

        # Predict the labels for the unlabelled data using Model 2
        y_train3_labels = pd.DataFrame(clf2.predict(X_train3_unlabelled))
        # Combine the labelled and predicted labels for training Model 3
        y_train3 = pd.concat([y_train3_labelled, y_train3_labels])
        # Train and predict using DecisionTreeRegressor for Model 3
        _, y_pred3 = regressor_DecisionTree(X_train1, X_test, y_train3)

    else:
        # Train and predict using DecisionTreeClassifier for Model 1
        clf1, y_pred1 = classifier_DecisionTree(X_train1, X_test, y_train1)

        # Train and predict using DecisionTreeClassifier for Model 2
        clf2, y_pred2 = classifier_DecisionTree(X_train2, X_test, y_train2)

        # Predict the labels for the unlabelled data using Model 2
        y_train3_unlabelled = clf2.predict(X_train3_unlabelled)
        # Combine the labelled and predicted labels for training Model 3
        X_train3 = pd.concat([X_train3_labelled, X_train3_unlabelled])
        y_train3 = pd.concat([y_train3_labelled, pd.Series(y_train3_unlabelled)])
        # Train and predict using DecisionTreeClassifier for Model 3
        _, y_pred3 = classifier_DecisionTree(X_train3, X_test, y_train3)

    # Evaluate the models using the appropriate metric
    err1 = evaluate_models(y_pred1, y_test, classf)
    err2 = evaluate_models(y_pred2, y_test, classf)
    err3 = evaluate_models(y_pred3, y_test, classf)

    return err1, err2, err3


def method_SSL(data, target ,classf=0):

    err1_list = []
    err2_list = []
    err3_list = []

    X = data.drop(target, axis=1)
    y = data[target]

    val = ""

    for i in range(26):

        print("Iteration: ", i)
        err1, err2, err3 = models_DT(X, y, classf)
        err1_list.append(err1)
        err2_list.append(err2)
        err3_list.append(err3)

    if classf == 0:
        print('---- Model 1 ----')
        print('MSE Score: ', round(sum(err1_list)/len(err1_list),5))
        print('---- Model 2 ----')
        print('MSE Score: ', round(sum(err2_list)/len(err2_list),5))
        print('---- Model 3 ----')
        print('MSE Score: ', round(sum(err3_list)/len(err3_list),5))
        print('-----------------')
        print('-----------------')
        if round(sum(err3_list)/len(err3_list),5) > round(sum(err2_list)/len(err2_list),5):
            print("No detect leakage!")
            return 0, val

        else:
            print("Detect leakage!")
            return 1, val

        print('-----------------')
        print('-----------------')


    else:
        print('---- Model 1 ----')
        print('F1 Score: ', round(sum(err1_list)/len(err1_list),5))
        print('---- Model 2 ----')
        print('F1 Score: ', round(sum(err2_list)/len(err2_list),5))
        print('---- Model 3 ----')
        print('F1 Score: ', round(sum(err3_list)/len(err3_list),5))
        print('-----------------')
        print('-----------------')
        if round(sum(err3_list)/len(err3_list),5) <= round(sum(err2_list)/len(err2_list),5):
            if round(sum(err3_list)/len(err3_list),5) == 1.0 == round(sum(err2_list)/len(err2_list),5) == 1.0:
                val = "overfitting"
            print("No detect leakage!")
            return 0, val

        else:
            print("Detect leakage!")
            return 1,val

        print('-----------------')
        print('-----------------')
