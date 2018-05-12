import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

df = pd.read_csv("../../data/transformed.csv")

vectoriser = TfidfVectorizer(min_df=0.01, max_df=0.7)
df2 = pd.DataFrame(list(vectoriser.fit_transform(df['lemmatized'].values.astype('U')).toarray()))
df = pd.concat([df, df2], axis=1)

del df['comment_text']
del df['without_stopwords']
del df['lemmatized']
columnCount = len(df.columns)

# print column names (numbers are for tf/idf)
print(list(df))

# print(df)

# ngram_vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w+\b', min_df=1)
# df['ngrams'] = list(ngram_vectorizer.fit_transform(df['lemmatized'].values.astype('U')).toarray())

# print(df.ngrams)

toxic = df.loc[df['toxic'] == 1]
nontoxic = df.loc[df['toxic'] == 0]
nontoxic = nontoxic.sample(n=len(toxic.index))

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(toxic.iloc[:, 2:columnCount], toxic["toxic"], test_size=0.2)
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(nontoxic.iloc[:, 2:columnCount], nontoxic["toxic"], test_size=0.2)
X_train = X_train_t.append(X_train_n)
X_test = X_test_t.append(X_test_n)
y_train = y_train_t.append(y_train_n)
y_test = y_test_t.append(y_test_n)

# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

clf = svm.SVC(C=1.0, cache_size=200, gamma='auto', kernel='rbf').fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f1_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
