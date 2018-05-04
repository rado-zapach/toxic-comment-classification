import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

df = pd.read_csv("../../data/transformed2.csv")

vectoriser = TfidfVectorizer(min_df=0.2)
df['tfidf'] = list(vectoriser.fit_transform(df['lemmatized'].values.astype('U')).toarray())

# print(df.tfidf)

# ngram_vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w+\b', min_df=1)
# df['ngrams'] = list(ngram_vectorizer.fit_transform(df['lemmatized'].values.astype('U')).toarray())

# print(df.ngrams)

toxic = df.loc[df['toxic'] == 1]
nontoxic = df.loc[df['toxic'] == 0]
nontoxic = nontoxic.sample(n=len(toxic.index))
df = pd.concat([toxic, nontoxic])

X_train, X_test, y_train, y_test = train_test_split(df['tfidf'].tolist(), df["toxic"].tolist(), test_size=0.2)

clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False).fit(X_train, y_train)

y_pred = clf.predict(X_test)

# skore = 0
# for j in range(0,len(test_vals)):
#     if(test_vals[j] == y_test[j]):
#         skore = skore + 1
#
# print(skore)
# print(len(test_vals))
# print(skore/len(test_vals))

# print(df.tfidf)

print(f1_score(y_test, y_pred, average="macro"))
print(precision_score(y_test, y_pred, average="macro"))
print(recall_score(y_test, y_pred, average="macro"))
print(accuracy_score(y_test, y_pred))
