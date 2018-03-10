import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data from csv
df = pd.read_csv('../data/train.csv')

# new column - joins all toxic columns as one boolean column
df['t'] = (df['toxic'] + df['severe_toxic'] + df['obscene'] + df['threat'] + df['insult'] + df['identity_hate']) > 0
# remove unnecessary columns
del df['id'], df['toxic'], df['severe_toxic'], df['obscene'], df['threat'], df['insult'], df['identity_hate']

# new column - number of words in text
df['words'] = df['comment_text'].str.split().str.len()
# new column - number uppercase letters in text
df['upper'] = df['comment_text'].str.count('[A-Z]')

# df[df.t].groupby("words").size()
# df[df.words == 1250]

# df2 = df[df.t]
# del df2['comment_text'], df2['t']

# df2.groupby('words').size().plot()
# df2.groupby('upper').size().plot()
# plt.show()

# df2.plot()
# df2.plot(x='words', y='upper', kind='scatter')
# plt.show()