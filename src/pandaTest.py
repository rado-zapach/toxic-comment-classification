import nltk
import pandas as pd
from sklearn.feature_extraction import text
from textblob import TextBlob
import time

start = time.time()

# load data from csv
df = pd.read_csv('../data/train.csv')

# new column - joins all toxic columns as one boolean column
df['toxicity'] = (df['toxic'] + df['severe_toxic'] + df['obscene'] + df['threat'] + df['insult'] + df['identity_hate']) > 0
# remove unnecessary columns
del df['id'], df['toxic'], df['severe_toxic'], df['obscene'], df['threat'], df['insult'], df['identity_hate']

# new column - number of words in text
df['words'] = df['comment_text'].str.split().str.len()

# new column - number of uppercase letters in text
df['upper'] = df['comment_text'].str.count('[A-Z]')

# sentiment
df[['polarity', 'subjectivity']] = df['comment_text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
print("sentiment done")

# remove stop words
stop = text.ENGLISH_STOP_WORDS
pat = r'\b(?:{})\b'.format('|'.join(stop))
df['without_stopwords'] = df['comment_text'].str.replace(pat, '')
df['without_stopwords'] = df['without_stopwords'].str.replace(r'\s+', ' ')
print("stop words done")

# lemmatization
nltk.download('wordnet')
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])


df['lemmatized'] = df.without_stopwords.apply(lemmatize_text)
print("lemmatization done")

end = time.time()
print(end - start)
