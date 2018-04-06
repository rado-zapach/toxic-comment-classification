import nltk
import pandas as pd
from sklearn.feature_extraction import text
from textblob import TextBlob
import time
import re

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])

def toxicity_to_int(toxicity):
    if toxicity:
        return 1
    else:
        return 0

def check_links(text):
    if re.search("(?P<url>https?://[^\s]+)", text) is not None:
        return 1
    else:
        return 0

def uppercase_normalize(text):
    return sum(1 for c in text if c.isupper()) / len(text)

start = time.time()

# load data from csv
df = pd.read_csv('../data/train.csv')

# new column - joins all toxic columns as one boolean column
df['toxicity'] = (df['toxic'] + df['severe_toxic'] + df['obscene'] + df['threat'] + df['insult'] + df['identity_hate']) > 0

# remove unnecessary columns
del df['id'], df['toxic'], df['severe_toxic'], df['obscene'], df['threat'], df['insult'], df['identity_hate']

# change boolean toxicity to 1/0
df['toxic'] = df.toxicity.apply(toxicity_to_int)
del df['toxicity']

# new column - number of words in text
df['words'] = df['comment_text'].str.split().str.len()
# normalize words count
max_words = df['words'].max()
df['words_normalized'] = df['words'] / max_words
del df['words']

# normalized upper case count
df['upper_normalized'] = df.comment_text.apply(uppercase_normalize)

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

df['lemmatized'] = df.without_stopwords.apply(lemmatize_text)
print("lemmatization done")

# check if comment has any link
df['has_link'] = df.comment_text.apply(check_links)

df.to_csv('../data/transformed.csv')

end = time.time()
print(end - start)
