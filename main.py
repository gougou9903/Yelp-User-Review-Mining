import json
import nltk
import string
import vectorization
import numpy as np
import lda
import gensim


from tokenization import tokenization
from vectorization import vectorize

def get_tokens():
	with open('data/chres_review.json') as reviewFile:
		for line in reviewFile:
				jLine = json.loads(line)
				perReview = jLine['text']
				print perReview
		
		


def modify(): #Tokenization, stop words removal and stemming
	with open('data/chres_review_modified.json','w+') as targetFile:
		with open('data/chres_review.json') as sourceFile:
			for line in sourceFile:
				jLine = json.loads(line)
				token = tokenization(jLine['text'])
				jLine['text'] = token
				targetFile.write(json.dumps(jLine)+'\n')
		
chresBuss = open ( 'data/chinese_restaurants.json' )
chresReview = open ('data/chres_review_modified.json')
ids = []	#business_id list for restaurants
for line in chresBuss:
	ids.append(json.loads(line)['business_id'])
	
print ids[0]

r_number = 0   #number of reviews per res
reviews = list()
for line in chresReview:
	jLine = json.loads(line)
	id = jLine['business_id']
	if id in ids[20]:
		r_number += 1
		str = " ".join(jLine['text'])
		reviews.append(str)

print "review numbers: {}".format(r_number)

reviews2 = list()   #split reviews into a list
for string in reviews:
	reviews2.append(string.split())
	
# print reviews2
dictionary = gensim.corpora.Dictionary(reviews2)
dictionary.save('dictionary.dict')
# print(dictionary)

corpus = [dictionary.doc2bow(review) for review in reviews2]
#print "corpus :", corpus
gensim.corpora.MmCorpus.serialize('corpus.mm', corpus)

tfidf = gensim.models.TfidfModel(corpus)

lda = gensim.models.ldamodel.LdaModel(corpus = corpus, id2word = dictionary, num_topics=10, update_every=0, passes=20)
topicArray =  lda.print_topics(10)

for i, topic in enumerate(topicArray):
	print('*Topic {}\n- {}'.format(i, topic))

print "distribution 0 : ", lda[corpus[0]]
print "distribution 1 : ", lda[corpus[1]]
print "distribution 2 : ", lda[corpus[2]]
print "distribution 3 : ", lda[corpus[3]]
print "distribution 4 : ", lda[corpus[4]]
print "distribution 5 : ", lda[corpus[5]]


# t = vectorize(reviews)
# feature_names = vectorization.tfidf.get_feature_names()
# 
# print t

# for col in t.nonzero()[1]:
# 	print feature_names[col], ' - ', t[2, col]
# 
# model = lda.LDA(n_topics=5, n_iter=1500, random_state=1)
# model.fit(t.toarray())

# gCorpus = gensim.matutils.Sparse2Corpus(t, documents_columns=True)
# print gCorpus

# print feature_names

# lda = gensim.models.ldamodel.LdaModel(corpus=gCorpus,num_topics=10)
# print lda.print_topics(5)


