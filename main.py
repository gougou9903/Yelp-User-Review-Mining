import json
import nltk
import string
import vectorization
import numpy as np
import lda
import gensim


from tokenization import tokenization
from vectorization import vectorize
from textblob import TextBlob


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
reviews = list()  # Modified reviews
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
print "corpus :", len(corpus)
gensim.corpora.MmCorpus.serialize('corpus.mm', corpus)

tfidf = gensim.models.TfidfModel(corpus)

lda = gensim.models.ldamodel.LdaModel(corpus = corpus, id2word = dictionary, num_topics=10, update_every=0, passes=20)
topicArray =  lda.print_topics(10)

print topicArray
# for i, topic in enumerate(topicArray):
# 	print('*Topic {}\n- {}'.format(i, topic))

print "distribution 0 : ", lda[corpus[0]] #test
print lda[corpus[0]][0][0]


#get the original reviews to do sentiment analysis

chresReviewOrig = open('data/chres_review.json')
reviewsOrig = list()
stars = list()
for line in chresReviewOrig:
	jLine = json.loads(line)
	id = jLine['business_id']
	if id in ids[20]:
		str = jLine['text']
		stars.append(jLine['stars'])
		reviewsOrig.append(str)
			
# blob0 = TextBlob(reviewsOrig[20])
# blob1 = TextBlob(reviewsOrig[21])
# blob2 = TextBlob(reviewsOrig[22])
# blob3 = TextBlob(reviewsOrig[23])
# blob4 = TextBlob(reviewsOrig[24])
# print "blob0: ", blob0.sentiment
# print "stars: ", stars[20]
# print "blob1: ", blob1.sentiment
# print "stars: ", stars[21]
# print "blob2: ", blob2.sentiment
# print "stars: ", stars[22]
# print "blob3: ", blob3.sentiment
# print "stars: ", stars[23]
# print "blob4: ", blob4.sentiment
# print "stars: ", stars[24]

scoreList = list()
for i, review in enumerate(reviewsOrig):
	blob = TextBlob(review)
	score = blob.sentiment.polarity * stars[i]
	scoreList.append(score)
	
print len(scoreList)
print "max: " , max(scoreList)
print "min: " , min(scoreList)

#get reviews for one specific topic
topic0Reviews = list()
for i,corp in enumerate(corpus):
	if lda[corp][0][0] == 0:
		print "topic0 reviews: ", reviewsOrig[i]


