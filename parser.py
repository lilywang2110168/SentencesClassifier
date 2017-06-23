import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import codecs
from sklearn import svm
from sklearn import naive_bayes
from sklearn.model_selection import (train_test_split, StratifiedKFold)
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)
from scipy import sparse
import numpy as np
import itertools

import os,glob


sentencesQuote = []


os.chdir("topics")
for file in glob.glob("*.data"):
    with codecs.open(file, "r", encoding="utf-8", errors='ignore') as f2:
        for line in f2:
            sentencesQuote.append(line)

print sentencesQuote
print len(sentencesQuote)


os.chdir("..")
with open('Wikipedia.txt', "r") as f1:
    data=f1.read().replace('\n', '')

sentencesPlot = sent_tokenize(data)

print len(sentencesPlot)

