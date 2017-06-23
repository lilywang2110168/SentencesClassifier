import nltk
from nltk.stem import WordNetLemmatizer
import codecs
from sklearn import svm
from sklearn import naive_bayes
from sklearn.model_selection import (train_test_split, StratifiedKFold)
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)
from scipy import sparse
import numpy as np
import itertools

wordnet_lemmatizer = WordNetLemmatizer()

sentencesPlot = []
sentencesQuote = []

# importing files
with open("plot.tok.gt9.5000") as f:
    for line in f:
        sentencesPlot.append(line)

with codecs.open("quote.tok.gt9.5000", "r", encoding="utf-8", errors='ignore') as f2:
    for line in f2:
        sentencesQuote.append(line)

# tokenizing
tokensPlot = [nltk.word_tokenize(sent) for sent in sentencesPlot]
tokensQuote = [nltk.word_tokenize(sent) for sent in sentencesQuote]
tokens = tokensPlot + tokensQuote

# constructing the dictionary
dictionary = {}
c = itertools.count()
for sent in tokens:
    for word in sent:
        lem = wordnet_lemmatizer.lemmatize(word, pos='v')
        lem = str(lem)
        if lem not in dictionary:
            dictionary[lem] = next(c)

# constructing the matrices
X = sparse.dok_matrix( (len(tokens),len(dictionary)) , dtype=np.int8 )
Y = np.zeros(len(tokens), dtype=np.int8)

for i, sent in enumerate(tokensPlot):
    for word in sent:
        lem = wordnet_lemmatizer.lemmatize(word, pos='v')
        lem = str(lem)
        index = dictionary[lem]
        X[i,index] = 1
    Y[i] = 1

for i, sent in enumerate(tokensQuote, start=len(tokensPlot)):
    for word in sent:
        lem = wordnet_lemmatizer.lemmatize(word, pos='v')
        lem = str(lem)
        index = dictionary[lem]
        X[i,index] = 1

# xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.20, random_state=42)

clf = naive_bayes.BernoulliNB()

# clf.fit(xtrain, ytrain)
# print 'train accuracy'
# print clf.score(xtrain, ytrain)
# print 'test accuracy'
# print clf.score(xtest, ytest)
#
# ypredict = clf.predict(xtest)
# print 'test confusion matrix'
# print confusion_matrix(ytest, ypredict)

skf = StratifiedKFold(n_splits=5)
confusion_matrices = []
accuracies = []
precisions = []
recalls = []
f1s = []
X = X.tocsr() # convert sparse matrix to a more efficient structure for slicing
for train,test in skf.split(X,Y):
    xtrain,xtest = X[train],X[test]
    ytrain,ytest = Y[train],Y[test]
    clf.fit(xtrain,ytrain)
    ypredict = clf.predict(xtest)
    confusion_matrices.append( confusion_matrix(ytest, ypredict) )
    accuracies.append( accuracy_score(ytest, ypredict) )
    precisions.append( precision_score(ytest,ypredict) )
    recalls.append( recall_score(ytest, ypredict) )
    f1s.append( f1_score(ytest, ypredict) )

print '5-fold cross-validation'
print 'sum of confusion matrices'
print sum(confusion_matrices)
print 'average accuracy'
print np.mean(accuracies)
print 'average precision'
print np.mean(precisions)
print 'average recall'
print np.mean(recalls)
print 'average f1'
print np.mean(f1s)
