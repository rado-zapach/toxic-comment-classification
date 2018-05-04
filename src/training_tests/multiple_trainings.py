import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def do_classification(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train.values, y_train.values)
    y_pred = clf.predict(X_test.values)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")

    print('accuracy, f1, precision, recall')
    print(str(round(accuracy, 2)))
    print(str(round(f1, 2)))
    print(str(round(precision, 2)))
    print(str(round(recall, 2)))
    print()


df = pd.read_csv("../data/transformed.csv", usecols=["polarity","subjectivity","has_link","upper_normalized","words_normalized","toxic"])

toxic = df.loc[df['toxic'] == 1]
nontoxic = df.loc[df['toxic'] == 0]
nontoxic = nontoxic.sample(n=len(toxic.index))
df = pd.concat([toxic, nontoxic])

X_train, X_test, y_train, y_test = train_test_split(df[["polarity","subjectivity","has_link","upper_normalized","words_normalized"]], df["toxic"], test_size=0.2)

# X_train, X_test, y_train, y_test = train_test_split(df[["polarity"]], df["toxic"], test_size=0.2)
# X_train, X_test, y_train, y_test = train_test_split(df[["subjectivity"]], df["toxic"], test_size=0.2)
# X_train, X_test, y_train, y_test = train_test_split(df[["has_link"]], df["toxic"], test_size=0.2)
# X_train, X_test, y_train, y_test = train_test_split(df[["upper_normalized"]], df["toxic"], test_size=0.2)
# X_train, X_test, y_train, y_test = train_test_split(df[["words_normalized"]], df["toxic"], test_size=0.2)

X_train, X_test, y_train, y_test = train_test_split(df[["has_link", "upper_normalized", "words_normalized"]], df["toxic"], test_size=0.2)

clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
do_classification(clf, X_train, X_test, y_train, y_test)

# clf = RidgeClassifier(tol=1e-2, solver="lsqr")
# do_classification(clf, X_train, X_test, y_train, y_test)
#
# clf = KNeighborsClassifier(n_neighbors=10)
# do_classification(clf, X_train, X_test, y_train, y_test)
#
# clf = RandomForestClassifier(n_estimators=100)
# do_classification(clf, X_train, X_test, y_train, y_test)
#
# clf = SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")
# do_classification(clf, X_train, X_test, y_train, y_test)
#
# clf = NearestCentroid()
# do_classification(clf, X_train, X_test, y_train, y_test)
#
# clf = MultinomialNB(alpha=.01)
# do_classification(clf, X_train, X_test, y_train, y_test)
#
# clf = DecisionTreeClassifier(max_depth=20)
# do_classification(clf, X_train, X_test, y_train, y_test)