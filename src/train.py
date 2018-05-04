import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm


df = pd.read_csv("../data/transformed2.csv", usecols=["polarity","subjectivity","has_link","upper_normalized","words_normalized","toxic"])
toxic = pd.DataFrame()
nontoxic = pd.DataFrame()

toxic =df.loc[df['toxic'] == 1]
#xx = len(numpy.flatnonzero(toxic["has_link"].values == 1))
nontoxic = df.loc[df['toxic'] == 0]
#xz = len(numpy.flatnonzero(nontoxic["has_link"].values == 1))

nontoxic = nontoxic.sample(n=16225)
data_zavislosti = toxic.append(nontoxic)

# Correction Matrix Plot
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
correlations = data_zavislosti[["polarity","subjectivity","has_link","upper_normalized","words_normalized"]].corr()
scatter_matrix(data_zavislosti[["polarity","subjectivity","has_link","upper_normalized","words_normalized"]])
plt.show()
print(correlations)
names = ['polarity', 'subjectivity', 'has_link', 'upper_normalized','words_normalized']
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,5,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()




from scipy.stats import chisquare
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data_zavislosti = toxic.append(nontoxic)
ch2, p = chisquare(data_zavislosti["has_link"])
print(ch2,p)
ch2, p = chisquare(data_zavislosti["polarity"])
print(ch2,p)
ch2, p = chisquare(data_zavislosti["subjectivity"])
print(ch2,p)
ch2, p = chisquare(data_zavislosti["upper_normalized"])
print(ch2,p)
ch2, p = chisquare(data_zavislosti["words_normalized"])
print(ch2,p)
#chi square,  pozor na pocet datv triedahc aby neboli rozdielne,vizualizovat feature,  ci je relevantna na zaklade
chi2_selector = SelectKBest(chi2, k=2)
X_kbest = chi2_selector.fit_transform(data_zavislosti[["subjectivity","has_link","upper_normalized","words_normalized"]].values, data_zavislosti["toxic"].values)

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(toxic[["polarity","subjectivity","has_link","upper_normalized","words_normalized"]], toxic["toxic"], test_size=0.2)
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(nontoxic[["polarity","subjectivity","has_link","upper_normalized","words_normalized"]], nontoxic["toxic"], test_size=0.2)

X_train = X_train_t.append(X_train_n)
X_test = X_test_t.append(X_test_n)
y_train = y_train_t.append(y_train_n)
y_test = y_test_t.append(y_test_n)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False).fit(X_train.values, y_train.values)

test_vals = clf.predict(X_test.values)
skore = 0
for j in range(0,len(test_vals)):
    if(test_vals[j] == y_test.values[j]):
        skore = skore + 1

print(skore)
print(len(test_vals))
print(skore/len(test_vals))
arr = numpy.flatnonzero(y_test.values == 0)
print(len(arr))
