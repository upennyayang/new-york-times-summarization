# This homework was done in pairs
# Teammember:   Yayang Tian: yaytian@seas.upenn.edu  24963298
#               Quan Dong :  qdong@seas.upenn.edu    18998424 

import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist, ConditionalFreqDist
import math
from nltk import NaiveBayesClassifier
import string

# Problem 1: Topic Classification

# 1.1 Coarse-Level Features
def get_coarse_level_features(dataset, output_file):
	# Import the corpus reader
	corpus_root = '/home1/c/cis530/data-hw2/'+dataset
	# Define the folder where the files are situated
	files_dataset = PlaintextCorpusReader(corpus_root, '.*')
	# Open the output_file
	output = open('/home1/c/cis530/data-hw2/'+output_file,'w')
	# Read the stopwlist
	stop_list = open('/home1/c/cis530/data-hw2/'+'stopwlist.txt').read()
	types_stop_list=stop_list.split()
	for fileid in files_dataset.fileids():
		# Output the docid
		output.write(dataset+'/'+fileid+' ')
		# Output the topic_name
		topic_name=fileid.split('/')[0]	
		output.write(topic_name+' ')
		# Output the num_tokens	
		tokens=files_dataset.words(fileid)
		output.write('tok:'+str(len(tokens))+' ')
		# Output the num_types
		types=set(tokens)
		output.write('typ:'+str(len(types))+' ')
		# Output the num_contents
		output.write('con:'+str(len([w for w in tokens if w not in types_stop_list]))+' ')
		# Output the num_sents
		sents = files_dataset.sents(fileid)
		output.write('sen:'+str(len(sents))+' ')
		# Output the avg_slen
		avg_slen=round(float(len(tokens))/float(len(sents)),2)
		output.write('len:'+str(avg_slen)+' ')
		# Output the num_caps
		output.write('cap:'+str(len([w for w in tokens if w[0]>='A' and w[0]<='Z'])))
		output.write('\n')
	output.close()

# test code:
# get_coarse_level_features('Training_set_xs', 'Training_set_xs.coarsefeatures')


# 1.2 Part-of-Speech Features
# 1.2.1 Data Preparation
	
def prepare_pos_features(Language_model_set, output_file):

	# Import the corpus reader
	corpus_root = '/home1/c/cis530/data-hw2/'+Language_model_set
	# Define the folder where the files are situated
	files_dataset = PlaintextCorpusReader(corpus_root, '.*')
	NOUNS=['NN','NNS','NP','NPS']
        VERBS=['VB','VBD','VBG','VBN','VBP','VBZ']
        ADJECTIVE=['JJ','JJR','JJS']
        ADVERB=['RB','RBR','RBS']
        PREPOSITION=['IN']
	pos_tag_tuple = nltk.pos_tag(files_dataset.words())
	# convert the tuple to list so that it can be changed
	pos_tag_list=[list(pos_tag_tuple[i]) for i in range(len(pos_tag_tuple))]
	# retag the POS
	for w in pos_tag_list:
		if w[1] in NOUNS:
			w[1]='NN'
		elif w[1] in VERBS:
			w[1]='VV'
		elif w[1] in ADJECTIVE:
			w[1]='ADJ'
		elif w[1] in ADVERB:
			w[1]='ADV'
		elif w[1] in PREPOSITION:
			w[1]='PREP'
	# invert it to tuple
	pos_tag_tuple_invert=[(t,w) for (w,t) in pos_tag_list]
	cfd=nltk.ConditionalFreqDist(pos_tag_tuple_invert)
	output = open('/home1/c/cis530/data-hw2/'+output_file,'w')
	for w in cfd['NN'].keys()[:200]:
                output.write('NN'+w+'\n')
	for w in cfd['VV'].keys()[:200]:
                output.write('VV'+w+'\n')
	for w in cfd['ADJ'].keys()[:200]:
                output.write('ADJ'+w+'\n')
	for w in cfd['ADV'].keys()[:100]:
                output.write('ADV'+w+'\n')
	for w in cfd['PREP'].keys()[:100]:
                output.write('PREP'+w+'\n')
	output.close()
	
# prepare_pos_features('Language_model_set', 'posfeatures_prepare')

# 1.2.2 Computing Features
def get_pos_features(dataset,feature_set_file,output_file):
	# Import the corpus reader
	corpus_root = '/home1/c/cis530/data-hw2/'+dataset
	# Define the folder where the files are situated
	files_dataset = PlaintextCorpusReader(corpus_root, '.*')
	feature_list = open('/home1/c/cis530/data-hw2/'+feature_set_file).read().split()
	output = open('/home1/c/cis530/data-hw2/'+output_file,'w')
	NOUNS=['NN','NNS','NP','NPS']
        VERBS=['VB','VBD','VBG','VBN','VBP','VBZ']
        ADJECTIVE=['JJ','JJR','JJS']
        ADVERB=['RB','RBR','RBS']
        PREPOSITION=['IN']	
	for fileid in files_dataset.fileids():
		# Output the docid
		output.write(dataset+'/'+fileid+' ')
		# Output the topic_name
		topic_name=fileid.split('/')[0]
		output.write(topic_name+' ')
		pos_tag_tuple = nltk.pos_tag(files_dataset.words(fileid))
		# convert the tuple to list so that it can be changed
		pos_tag_list=[list(pos_tag_tuple[i]) for i in range(len(pos_tag_tuple))]
		# retag the POS and replace the word with tag_word
		for w in pos_tag_list:
			if w[1] in NOUNS:
				w[0]='NN'+w[0]
				w[1]='NN'
			elif w[1] in VERBS:
				w[0]='VV'+w[0]
				w[1]='VV'
			elif w[1] in ADJECTIVE:
				w[0]='ADJ'+w[0]
				w[1]='ADJ'
			elif w[1] in ADVERB:
				w[0]='ADV'+w[0]
				w[1]='ADV'
			elif w[1] in PREPOSITION:
				w[0]='PREP'+w[0]
				w[1]='PREP'
		tw_list=[w[0] for w in pos_tag_list]		
		fd=FreqDist(tw_list)
		for tw in feature_list:
                        output.write(tw+':'+str(fd[tw])+' ')
		output.write('\n')
	output.close()

# get_pos_features('Training_set_xs','posfeatures_prepare','Training_set_xs.posfeatures')
		
# 1.3 Language Model Features

class BigramModel:
        category_root=[]
        files_dataset_category=[]
        word_list=[]
        bigram=[]
        fd = []
        cfd = []
        def __init__(self,category,corpus_root):
                self.category_root=[]
                self.files_dataset_category=[]
                self.word_list=[]
                self.bigram=[]
                self.fd = []
                self.cfd = []
                self.category_root=corpus_root+'/'+category
                self.files_dataset_category=PlaintextCorpusReader(self.category_root,'.*')
                self.word_list = self.files_dataset_category.words()
                self.bigram = nltk.bigrams(self.word_list)
                self.fd = FreqDist(self.word_list)
                self.cfd = nltk.ConditionalFreqDist(self.bigram)
        def get_prob_and_per(self,word_list):
                # The function takes a word_list and return both the log probability and log perplexity under the language model 
                n_types = len(set(word_list))
                n_tokens=len(word_list)
                # Calculate Log Prob with Laplace smoothing.
                log_prob = math.log(self.fd[word_list[0]]+1)-math.log(n_tokens+n_types)  #initializing prob for the first word
                for (w1,w2) in nltk.bigrams(word_list):
                    log_prob = log_prob+math.log(self.cfd[w1][w2]+1)-math.log(len(self.cfd[w1].keys())+n_types)
                # Calculate Log Perplexity
                log_per=float(1)/float(-n_tokens)*log_prob
                return log_prob, log_per
		
		
def get_lm_features(dataset,output_file):
        # Import the corpus reader
	corpus_root = '/home1/c/cis530/data-hw2/'+dataset
	# Define the folder where the files are situated
	files_dataset = PlaintextCorpusReader(corpus_root, '.*')	
        fin_model = BigramModel('Finance',corpus_root)
        hel_model = BigramModel('Health',corpus_root)
        res_model = BigramModel('Computers_and_the_Internet',corpus_root)
        co_model = BigramModel('Research',corpus_root)
        output = open('/home1/c/cis530/data-hw2/'+output_file,'w')
        for fileid in files_dataset.fileids():
		# Output the docid
		output.write(dataset+'/'+fileid+' ')
		# Output the topic_name
		topic_name=fileid.split('/')[0]
		output.write(topic_name+' ')		
		word_list = files_dataset.words(fileid)
		finprob,finper = fin_model.get_prob_and_per(word_list)		
		hlprob,hlper = hel_model.get_prob_and_per(word_list)	
		resprob,resper = res_model.get_prob_and_per(word_list)
		coprob,coper = co_model.get_prob_and_per(word_list)
		output.write('finprob:'+str(round(finprob,1))+' ')
		output.write('hlprob:'+str(round(hlprob,1))+' ')
		output.write('resprob:'+str(round(resprob,1))+' ')
		output.write('coprob:'+str(round(coprob,1))+' ')
		output.write('finper:'+str(round(finper,1))+' ')
		output.write('hlper:'+str(round(hlper,1))+' ')
		output.write('resper:'+str(round(resper,1))+' ')
		output.write('coper:'+str(round(coper,1))+' ')
		output.write('\n')
        output.close()

# get_lm_features('Training_set_xs','Training_set_xs.lmfeatures')


# 1.4 Combining Features
def combine_features(feature_files, output_file):
        output = open('/home1/c/cis530/data-hw2/'+output_file,'w')
        feature=[0]*len(feature_files)
        for i in range(len(feature_files)):
                docid_topic_feature=open('/home1/c/cis530/data-hw2/'+feature_files[i]).read().split('\n')
                docid_topic_feature_split=[docid_topic_feature[j].split()for j in range(len(docid_topic_feature))]
                feature[i]=[docid_topic_feature_split[j][2:]for j in range(len(docid_topic_feature_split))]
        for i in range(len(feature[0])):
                output.write(docid_topic_feature[i])
                for j in range(len(feature_files)-1):
                        output.write(' '.join(feature[j][i])+' ')
                output.write('\n')
        output.close()

# combine_features(['Training_set_xs.coarsefeatures','Training_set_xs.lmfeatures'], 'combine.features')
# combine_features(['Training_set_xs.coarsefeatures'], 'combine.features')
# combine_features(['Training_set_xs.coarsefeatures','Training_set_xs.lmfeatures','Training_set_xs.posfeatures'], 'combine.features')

        
# 1.5 Classifying Topics
def get_NB_classifier(training_examples):
        # Convert the training_examples to [['id','topic','feature1:f1','feature2:f2'...],...]
        id_topic_features_list=[list(id_topic_features.split()) for id_topic_features in open('/home1/c/cis530/data-hw2/'+training_examples).read().split('\n')]
        # Convert the above to [({'feature1':f1,'feature2':f2,...},'topic'),...]
        features_set=[({i[k+2].split(':')[0]:string.atof(i[k+2].split(':')[1]) for k in range(len(a[0])-2)},i[1]) for i in a]
        # Above sentence has syntax error on python 2.6.6, but works well on 2.7.2, I don't know why...
        classifier = nltk.NaiveBayesClassifier.train(features_set)
        return classifier
        
        

def classify_documents(test_examples, model, classifier_output):
        # Convert the test_examples to [['id','topic','feature1:f1','feature2:f2'...],...]
        id_topic_features_list=[list(id_topic_features.split()) for id_topic_features in open('/home1/c/cis530/data-hw2/'+test_examples).read().split('\n')]
        # Convert the above to [({'feature1':f1,'feature2':f2,...},'topic'),...]
        features_set=[({i[k+2].split(':')[0]:string.atof(i[k+2].split(':')[1]) for k in range(len(a[0])-2)},i[1]) for i in a]
        # Above sentence has syntax error on python 2.6.6, but works well on 2.7.2, I don't know why...
        
        id_and_true_topic_list=[id_topic_features[:2] for id_topic_features in id_topic_features_list]
        output = open('/home1/c/cis530/data-hw2/'+classifier_output,'w')
        for i in range(len(features_set)):
                predicted_topic=model.classify(features_set[i][0])
                output.write(id_and_true_topic_list[i][0]+' '+id_and_true_topic_list[i][1]+' ')
                output.write(predicted_topic)
                output.write('\n')
        output.close()
                                
        
# 2.1 Fit of a Word
def get_fit_for_word(sentence,word,langmodel):
        corpus_root='/home1/c/cis530/data-hw2/Language_model_set/'
        model=[]
        model=BigramModel(langmodel,corpus_root)     
        sent_split=sentence.split()
        # replace the blank by the word
        for i in range(len(sent_split)):
                if sent_split[i]=='<blank>':
                        sent_split[i]=word
        prob,per=model.get_prob_and_per(sent_split)
        return prob

# get_fit_for_word('Stocks <blank> this morning.','walked','Health')

# 2.2 Best Fit for a Sentence

def get_bestfit_topic(sentence, wordlist, topic):
        wordlist_prob = [[word,get_fit_for_word(sentence,word,topic)] for word in wordlist]
        wordlist_prob_dict=dict(wordlist_prob)
        b=wordlist_prob_dict.keys()
        n=b[0]#biggest prob word
        d=wordlist_prob_dict.values()
        m=d[0]#biggest prob of that word
        for name,value in wordlist_prob_dict.items() : 
           if value>m:
             m=value
             n=name
        return n

# get_bestfit_topic('a <blank> apple', ['big','black'], 'Research')
