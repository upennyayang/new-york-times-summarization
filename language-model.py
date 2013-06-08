# This homework was done in pairs
# Teammember: Yayang Tian:	yaytian@seas.upenn.edu 	24963298
#              Quan Dong :	qdong@seas.upenn.edu 	18998424 
#			  
			
from __future__ import division
import nltk
from nltk import FreqDist, ConditionalFreqDist
#import matplotlib.pyplot
import random
from nltk.corpus import brown
# Problem 1 Counting Words

# 1.1 Probability Distribution of Words
# a)
def get_prob_word_in_category(word, category=''):
	# returns the probability of the given word appearing in the given category 
	# (or the entire corpus, if no category is given).
	if category=='':
		text=brown.words() # get the text from the entire corpus
	else:
		text=brown.words(categories=category) # get the text from the given category
	return text.count(word)/len(text) 
	
# 1.2 Types and Tokens
# a)
def get_vocabulary_size(category=''):
	# returns the size of the vocabulary for a single category from the corpus. 
	# If no category is given, the function should return the vocabulary size for the entire corpus.
	if category=='':
		text=brown.words() # get the text from the entire corpus
	else:
		text=brown.words(categories=category) # get the text from the given category
	return len(set(text))

# b)
def get_type_token_ratio(category=''):
	# returns the type to token ratio for the given topic
	if category=='':
		text=brown.words() # get the text from the entire corpus
	else:
		text=brown.words(categories=category) # get the text from the given category
	return len(set(text))/len(text)
	
# 1.3 Word Frequency
# a)
def get_top_n_words(n, category=''):
	#return the most frequent n words from a category (or the entire corpus)
	if category=='':
		text=brown.words() # get the text from the entire corpus
	else:
		text=brown.words(categories=category) # get the text from the given category
	fdist=FreqDist(text)
	top_words=fdist.keys()
	return top_words[:n]
	
def get_bottom_n_words(n, category=''):
	#return the least frequent n words from a category (or the entire corpus)
	#return the most frequent n words from a category (or the entire corpus)
	if category=='':
		text=brown.words() # get the text from the entire corpus
	else:
		text=brown.words(categories=category) # get the text from the given category
	fdist=FreqDist(text)
	top_words=fdist.keys()
	return top_words[-n:]	

# b)
# The top 30 words: ['the', ',', '.', 'of', 'and', 'to', 'a', 'in', 'that', 'is', 'was', 'for', '``', "''", 'The', 'with', 'it', 'as', 'he', 'his', 'on', 'be', ';', 'I', 'by', 'had', 'at', '?', 'not', 'are']
# The bottom 30 words: ["you's", "you'uns", 'youngish', "youngster's", 'younguh', 'ys', 'yucca', 'yuse', 'yyyy', 'zealot', 'zebra', 'zenith', 'zeroed', 'zeros', 'zip', 'zipped', 'zipper', 'zlotys', 'zombie', 'zombies', 'zoned', 'zoology', 'zoomed', 'zooming', 'zooms', 'zoooop', 'zorrillas', 'zu', 'zur', '{0,T}']
# What do you notice? Why do you think this is?

# c)
def plot_word_counts():
	# produces a word frequency histogram for the news category of the corpus
	text=brown.words(categories='news')
	fdist=FreqDist(text)
	x=[fdist[w] for w in fdist]
	# Prepare the histogram
	matplotlib.pyplot.hist(x,bins=3000)
	# Annotate the graph
	matplotlib.pyplot.xlabel('Number of occurences of a word')
	matplotlib.pyplot.ylabel('Number of words that appear that many times')
	matplotlib.pyplot.title('Number of words by number of occurences')
	# Set the axis ([Min X, Max X, Min Y, Max Y])
	matplotlib.pyplot.axis([0,500,0,500])
	
	matplotlib.pyplot.show()


	
# Problem 2 Context and Similarity

# 2.1 Word Contexts
# a)
def get_word_contexts(word):
	#returns each context for the given word in the news category of the corpus
	text=brown.words(categories='news')	
	contexts=[(text[i],text[i+2]) for i in range(len(text)-2) if text[i+1]==word]
	return list(set(contexts))

# b)
def get_common_contexts(word1, word2):
	#returns the unique contexts shared by word1 and word2 in the news category of the corpus
	contexts1=get_word_contexts(word1)
	contexts2=get_word_contexts(word2)
	common_contexts=contexts1 and contexts2
	return common_contexts
	
# 2.2 Measuring Similarity
# a)
def create_feature_space(sentence_list):
	# create a Python dict mapping each unique word type in all of the sentences to a consecutive integer starting from zero (order doesn't matter). 
	# This creates a mapping between each word and the element in each vector that will represent it.
	joint_sentences=' '.join(sentence_list)
	split_words=joint_sentences.split()
	word_type=list(set(split_words))
	dict_map=[(word_type[i],i) for i in range(len(word_type))]
	return dict(dict_map)	

def vectorize(feature_space, sentences):
        k=feature_space.keys()
        s_list=sentences.split()
        return [val in s_list for val in k]        #If a value is in feature_space.keys(), return True, else return False

		

# 2.2 
# b)        
def dice_similarity(X,Y):
        up=sum(min(X[i],Y[i]) for i in range(len(X)))
        dn=sum((X[i]+Y[i]) for i in range(len(X)))
        return 2*up/dn      
        
def jaccard_similarity(X,Y):
        up=sum(min(X[i],Y[i]) for i in range(len(X)))
        dn=sum(max(X[i],Y[i]) for i in range(len(X)))
        return up/dn

def cosine_similarity(X,Y):
        up=sum((X[i]*Y[i]) for i in range(len(X)))
        d1=sum((X[i]*X[i]) for i in range(len(X)))
        d2=sum((Y[i]*Y[i]) for i in range(len(X)))
        return up/(d1*d2)**0.5

#3.1 A Sliding Window

def make_ngram_tuples(samples, n):
	if n==1:
		ngram_tuples=[(None,samples[i]) for i in range(len(samples))]
	else:
		ngram_tuples=[(tuple(samples[i:i+n-1]),samples[i+n-1]) for i in range(len(samples)-n+1)]
	return ngram_tuples

#3.2 Building Language Models
class NGramModel:

	def __init__(self,training_data,n):
		self.data=training_data
		self.n=n
		self.ngram=make_ngram_tuples(training_data, n)

	def prob(self,context, event):
		all_contexts=[self.ngram[i][0] for i in range(len(self.ngram))] # all the contexts in the model
		fdist_contexts=FreqDist(all_contexts)
		fdist_ngram=FreqDist(self.ngram)
		conditional_prob=fdist_ngram[(context,event)]/fdist_contexts[context] # P(event|context)=P(context,event)/P(context)
		return conditional_prob
#3.3 Generating Text	
	def generate(self,n,context):
	# Generates a sentence of length upto n words starting with thegiven context. 
	# For the unigram language model, only None will be provided as the context. 
	# Context will be supplied as a tuple, of length 1 for bigrams and 2 for trigram model.	
		word_types=list(set(self.data)) # List for word types in training_data
		generating_text=list(context) #Initialize the result by the context(the first word)
		for j in range(n): # To generate more than n words
			v=[word_types[i] for i in range(len(word_types)) if self.prob(context,word_types[i])>0] # for all type vi whose conditional Prob against context is higher than 0
			r=random.random() #uniformly random number
			s=0
			for k in range(len(v)): #for each vi
				if s<=r and r<(s+self.prob(context,v[k])): #If s<=r<=s+P(v|context), pick word and break
					generating_text.append(v[k]) #update result
					context=(v[k],) #update new context
					break
				else: # else increase s by P(v|context)
					s=s+self.prob(context,v[k])
		return generating_text
		
	

