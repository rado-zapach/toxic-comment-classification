import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer

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
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print('accuracy, f1, precision, recall')
    print(str(round(accuracy, 2)))
    print(str(round(f1, 2)))
    print(str(round(precision, 2)))
    print(str(round(recall, 2)))
    print()


df = pd.read_csv("../../data/transformed.csv")

# retrieve tfidf frequencies
vectoriser = TfidfVectorizer(min_df=0.01, max_df=0.7)
# put tfidf to new dataframe for easier work
df2 = pd.DataFrame(list(vectoriser.fit_transform(df['lemmatized'].values.astype('U')).toarray()))
# add tfidf attributes to existing attributes
df = pd.concat([df, df2], axis=1)

# remove unnecesery columns
del df['comment_text']
del df['without_stopwords']
del df['lemmatized']
# column count - all columns from col. 2 are attributes
# cols 7 and more are tfidf
columnCount = len(df.columns)

# print column names (numbers are for tf/idf)
print(list(df))

toxic = df.loc[df['toxic'] == 1]
nontoxic = df.loc[df['toxic'] == 0]
nontoxic = nontoxic.sample(n=len(toxic.index))
df = pd.concat([toxic, nontoxic])

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(toxic.iloc[:, 2:columnCount], toxic["toxic"], test_size=0.2)
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(nontoxic.iloc[:, 2:columnCount], nontoxic["toxic"], test_size=0.2)
X_train = X_train_t.append(X_train_n)
X_test = X_test_t.append(X_test_n)
y_train = y_train_t.append(y_train_n)
y_test = y_test_t.append(y_test_n)

clf = svm.SVC(C=1.0, cache_size=200, gamma='auto', kernel='rbf').fit(X_train, y_train)
do_classification(clf, X_train, X_test, y_train, y_test)

# clf = RandomForestClassifier(n_estimators=100)
# do_classification(clf, X_train, X_test, y_train, y_test)

# clf = DecisionTreeClassifier(max_depth=20)
# do_classification(clf, X_train, X_test, y_train, y_test)