
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy.matlib as np
import codecs
from sklearn import svm
from sklearn.model_selection import train_test_split


wordnet_lemmatizer = WordNetLemmatizer()

sentencesPlot=[]
sentencesQuote=[]

#importing files
with open ("plot.tok.gt9.5000") as f: 
	for line in f:
		sentencesPlot.append(line)

with codecs.open("quote.tok.gt9.5000","r", encoding="utf-8", errors='ignore') as f2:
	for line in f2:
		sentencesQuote.append(line)



#tokenizing
tokensPlot = [nltk.word_tokenize(sent) for sent in sentencesPlot]
tokensQuote = [nltk.word_tokenize(sent) for sent in sentencesQuote]

tokens=tokensPlot+tokensQuote

#constructing the dictionary
dictionary= {}
i=0
for sent in tokens:
	for word in sent:
		lem=wordnet_lemmatizer.lemmatize(word,pos='v')
		lem=str(lem)
		if lem not in dictionary:
			dictionary[lem]=i
			i=i+1

length = len(dictionary)

#constructing the matrices

X=[]
Y=[]
for sent in tokensPlot:
	for word in sent: 
		Z=[0] * length
		lem=wordnet_lemmatizer.lemmatize(word,pos='v')
		index=dictionary[lem]
		#print index
		Z[index]=1

	X.append(Z)
	Y.append(1)


for sent in tokensQuote:
	for word in sent: 
		Z=[0] * length
		lem=wordnet_lemmatizer.lemmatize(word,pos='v')
		index=dictionary[lem]
		#print index
		Z[index]=1
	X.append(Z)
	Y.append(0)


xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.20,random_state=42)

print "I am here"

clf = svm.SVC()

clf.fit(X, Y)
print clf.score(xtrain, ytrain)
print clf.score(xtest, ytest)  

#print "Now I am here"
#or arr in X:
	#print clf.predict([arr])










#print len(dictionary)




