# Importing the libraries
# import numpy as np
# import matplotlib.plot as plt
import pandas as pd

# import sklearn


def log(header: str, cargo: object):
    print(header)
    print(cargo)
    print("---------------------------------------")


def runone():
    # Importing the dataset
    dataset = pd.read_csv('Social_Network_Ads.csv')
    log("dataset", dataset)

    X = dataset.loc[:, ['Age', 'EstimatedSalary']].values
    y = dataset.loc[:, 'Purchased'].values
    log("X", X)
    log("y", y)

    # Encoding labels for y
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    X[:, 0] = le.fit_transform(X[:, 0])
    log("X (encoded)", X)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    log("X_train", X_train)
    log("X_test", X_test)
    log("y_train", y_train)
    log("y_test", y_test)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    log("sc", sc)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    log("X_train", X_train)
    log("X_test", X_test)

    # Training the Naive Bayes model on the Training set
    from sklearn.naive_bayes import GaussianNB

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    log("classifier", classifier)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # log("y_pred", y_pred)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score

    ac = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    log("ac", ac)
    log("cm", cm)

if __name__ == '__main__':
    runone()
