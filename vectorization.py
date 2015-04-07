import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf = True)

def vectorize(list):
	
	
	tfs = tfidf.fit_transform( list )
	return tfs