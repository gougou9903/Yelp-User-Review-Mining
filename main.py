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

final_list = []
final_star = []

# for loop_res in range(len(ids)):
# 	print "loop_res: ", loop_res
# 	chresBuss.seek(0)
# 	for line in chresBuss:
# 		jLine = json.loads(line)
# 		final_star.append(jLine['stars'])

for loop_res in range(len(ids)):
	print "loop_res: ", loop_res
	r_number = 0   #number of reviews per res
	reviews = list()  # Modified reviews
	chresReview.seek(0)
	for line in chresReview:
		jLine = json.loads(line)
		id = jLine['business_id']   #id in review
		if id in ids[loop_res]:    #---------------------->  restuarant id!!!
			r_number += 1
			str = " ".join(jLine['text'])
			reviews.append(str)
	
	print "reviews: ", reviews
	if r_number == 0: 
		continue

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

	for i, topic in enumerate(topicArray):
		print('*Topic {}\n- {}'.format(i, topic))

	# print "distribution 0 : ", len(lda[corpus[0]]) #test
	# print lda[corpus[0]][0][0]


	#get the original reviews to do sentiment analysis

	chresReviewOrig = open('data/chres_review.json')
	reviewsOrig = list()
	stars = list()
	for line in chresReviewOrig:
		jLine = json.loads(line)
		id = jLine['business_id']
		if id in ids[loop_res]:         #-------------------> restaurant id!!!!
			str = jLine['text']
			stars.append(jLine['stars'])
			reviewsOrig.append(str)
				

	#get reviews for one specific topic
	topicReviews = [[] for i in range(10)]
	for i,corp in enumerate(corpus):
		for distribution in lda[corp]:
			print "lda distribution: ", distribution

			if distribution[0] == 0:
				topicReviews[0].append(reviewsOrig[i])
			if distribution[0] == 1:
				topicReviews[1].append(reviewsOrig[i])
			if distribution[0] == 2:
				topicReviews[2].append(reviewsOrig[i])
			if distribution[0] == 3:
				topicReviews[3].append(reviewsOrig[i])
			if distribution[0] == 4:
				topicReviews[4].append(reviewsOrig[i])
			if distribution[0] == 5:
				topicReviews[5].append(reviewsOrig[i])
			if distribution[0] == 6:
				topicReviews[6].append(reviewsOrig[i])
			if distribution[0] == 7:
				topicReviews[7].append(reviewsOrig[i])
			if distribution[0] == 8:
				topicReviews[8].append(reviewsOrig[i])
			if distribution[0] == 9:
				topicReviews[9].append(reviewsOrig[i])

	scoreList = [[] for i in range(10)]

	for n in range(10):
		for i, review in enumerate(topicReviews[n]):
			print "topicReviews{} length: {}".format(n,len(topicReviews[n]))
			blob = TextBlob(review)
			score = blob.sentiment.polarity * stars[i]
			scoreList[n].append(score)

	print "scoreList: ", scoreList

	for i,s in enumerate(scoreList):
		if s == []:
			continue


		max_score = max(s)
		min_score = min(s)
		scaledList = []
		print "max: " , max_score
		print "min: " , min_score

		if max_score == min_score:
			final_list.append(-1)
			print "topic{} dismiss.".format(i)
			print "*************************"
			continue

		for j in s:
			scaled_score = (j - min_score)/(max_score - min_score) * 5
			scaledList.append(scaled_score)

		#print "scaledList: ", scaledList
		# with open('data/output.txt','w+') as text_file:

		# 	text_file.write("topic: {} ; ".format(i))
		# 	text_file.write("score: {} | ".format(np.mean(scaledList)))
			
		text_file = open("output.txt", "w+")
		text_file.write("topic: {} ;".format(i))
		text_file.write("score: {} | ".format(np.mean(scaledList)))
		text_file.close()

		
		final_list.append(np.mean(scaledList))
		

		print "topic: ", i
		print "numbers: ", len(s)
		print "max: " , max_score
		print "min: " , min_score
		print "scaled score: ", np.mean(scaledList)
		print "*************************"

# print "final_list", final_list
# text_file = open("output.txt", "w+")
# for item in final_star:
# 	text_file.write("{}, ".format(item))
# text_file.close()

