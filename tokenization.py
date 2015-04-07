import json
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def tokenization(str):
	lowers = str.lower()
	remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
	no_punctuation = lowers.translate(remove_punctuation_map)
	tokens = nltk.word_tokenize(no_punctuation)
	filtered = [w for w in tokens if not w in stopwords.words('english')]
	
	wordList2 = list()
	for word in filtered:
		word2 = stemmer.stem(word)
		wordList2.append(word2)
	
	return wordList2