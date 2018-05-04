import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# load data from csv
df = pd.read_csv('../data/transformed2.csv')

v = TfidfVectorizer()
x = v.fit_transform(df['lemmatized'].values.astype('U'))

df['tfidf']=list(x)