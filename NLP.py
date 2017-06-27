import nltk
from nltk.stem import WordNetLemmatizer
import codecs
from sklearn import svm
from sklearn import naive_bayes
from sklearn.model_selection import (train_test_split, StratifiedKFold)
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)
from sklearn.externals import joblib
from scipy import sparse
import numpy as np
import itertools
import os
import glob
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--save-file', dest='save_file')
args = parser.parse_args()

wordnet_lemmatizer = WordNetLemmatizer()

sentencesPlot = []
sentencesQuote = []

# importing files
with open("plot.tok.gt9.5000") as f:
    for line in f:
        sentencesPlot.append(line)
'''
with codecs.open("quote.tok.gt9.5000", "r", encoding="utf-8", errors='ignore') as f2:
    for line in f2:
        sentencesQuote.append(line)


'''
sentencesQuote = []


os.chdir("topics")
for file in glob.glob("*.data"):
    with codecs.open(file, "r", encoding="utf-8", errors='ignore') as f2:
        for line in f2:
            sentencesQuote.append(line)

os.chdir("..")

#print sentencesQuote
#print len(sentencesQuote)
'''

os.chdir("..")
with codecs.open('Wikipedia.txt', "r",encoding="utf-8", errors='ignore') as f1:
    data=f1.read().replace('\n', '')

sentencesPlot = nltk.sent_tokenize(data)

print len(sentencesPlot)
'''


# tokenizing
tokensPlot = [nltk.word_tokenize(sent) for sent in sentencesPlot]
#print tokensPlot
tokensQuote = [nltk.word_tokenize(sent) for sent in sentencesQuote]
#print tokensQuote
tokens = tokensPlot + tokensQuote



#preprocessing
for i in range(len(tokens)): 
    for j in range(len(tokens[i])):
        tmp=wordnet_lemmatizer.lemmatize(tokens[i][j], pos='v')
        tmp=tmp.encode('ascii', 'ignore')
        tmp=tmp.lower()
        
        if tmp.isdigit():
            tmp='0'

        tokens[i][j]=tmp


# constructing the dictionary
dictionary = {}
c = itertools.count()
for sent in tokens:
    for word in sent:
        if word not in dictionary:
            dictionary[word] = next(c)

#print dictionary

# constructing the matrices
X = sparse.dok_matrix( (len(tokens),len(dictionary)) , dtype=np.int8 )
Y = np.zeros(len(tokens), dtype=np.int8)

for i, sent in enumerate(tokensPlot):
    for word in sent:
        index = dictionary[word]
        X[i,index] = 1
    Y[i] = 1

for i, sent in enumerate(tokensQuote, start=len(tokensPlot)):
    for word in sent:
        index = dictionary[word]
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
    f1s.append( f1_score(ytest, ypredict))



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

if args.save_file:
    final_classifier = naive_bayes.BernoulliNB()
    final_classifier.fit(X,Y)
    with open(args.save_file, 'w+') as f:
        joblib.dump(final_classifier, f)
    print 'Saved trained classifier at {}'.format(os.path.abspath(args.save_file))


sentences=[]



#Using the trained model to classify reviews (by sentence)


#importing the file
with codecs.open('LaptopReviews.txt', "r",encoding="utf-8", errors='ignore') as f1:
     for line in f1:
        sents=nltk.sent_tokenize(line)
        for sent in sents:
            sentences.append(sent)

#tokenizing
tokensReview = [nltk.word_tokenize(sent) for sent in sentences]


#preoprocessing
for i in range(len(tokensReview)): 
    for j in range(len(tokensReview[i])):
        tmp=wordnet_lemmatizer.lemmatize(tokensReview[i][j], pos='v')
        tmp=tmp.encode('ascii', 'ignore')
        tmp=tmp.lower()
        tokensReview[i][j]=tmp


#contructing the matrix
X = sparse.dok_matrix( (len(tokensReview),len(dictionary)) , dtype=np.int8 )
for i, sent in enumerate(tokensReview):
    for word in sent:
        if word in dictionary: 
            index = dictionary[word]
            X[i,index] = 1




#predicting and writing results to files
fileObj = open('ObjReviews2.txt','w')
fileSub= open('SubReviews2.txt','w')


ypredict = clf.predict(X)

for i in range(len(ypredict)): 
    if ypredict[i]==0:
        fileObj.write(sentences[i]+'\n')
    else:
        fileSub.write(sentences[i]+'\n')


print "Done!"




































