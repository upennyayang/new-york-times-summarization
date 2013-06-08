#---------------------------------------------------------------#
#   CIS 530 Homework 3			                        #
#   Yayang Tian	        24963298        yaytian@seas.upenn.edu  #
#   Quan Dong	        18998424        qdong@seas.upenn.edu    #
#---------------------------------------------------------------#

import nltk
import string
import math
import os

from stanford_parser.parser import Parser
from operator import itemgetter



# 1 Topic Words and Syntactic Dependencies

# 1.2 Topic Words
# a)
def load_topic_words(topic_file):  
    topic_words_list_string=[(line.split()) for line in open(topic_file).readlines()]
    topic_words_list_float=[[word,round(string.atof(score),4)] for [word,score]in topic_words_list_string]
    return dict(topic_words_list_float)

# load_topic_words('d30006t.ts')

# b)
def get_topic_words(topic_file):     #the same as load_topic_word. get topic sigature score dict
    topic_words_list_string=[(line.split()) for line in open(topic_file).readlines()]
    topic_words_list_float=[[word,round(string.atof(score),4)] for [word,score]in topic_words_list_string]
    return dict(topic_words_list_float)

def get_top_n_topic_words(topic_words_dict,n):
    topic_words_sorted=sorted(topic_words_dict,key=topic_words_dict.__getitem__, reverse=True)
    return topic_words_sorted[:n]

# tw = get_topic_words('d30006t.ts')
# get_top_n_topic_words(tw, 5)

# c)
def filter_top_n_words(topic_words_dict,n, word_list):
    unique_intersection = [word for word in word_list if word in set(topic_words_dict.keys())]
    n = max(n, len(unique_intersection)) #If there are fewer than n words in the intersection, return them all.
    unique_topic_words_sorted=sorted(unique_intersection,key=topic_words_dict.__getitem__, reverse=True)
    return unique_topic_words_sorted[:n]

# tw = get_topic_words('/path/to/topics.ts')
# filter_top_n_words(tw, 3, ['players', 'national', 'the', 'put', 'lawyer', 'a'])

# 1.4 Performing Parsing
# a)
def load_file_sentences(path):
    input_file = open(path)
    sent_list = nltk.sent_tokenize(input_file.read().replace('\n','').lower())#Convert all sentences to lowercase.
    input_file.close()
    return sent_list

# [debug] load_file_sentences('/home1/c/cis530/data-hw3/articles/d30006t/APW19981018.0836.clean')

def load_collection_sentences(path,n=None):
    fileid_list=os.listdir(path)
    if n==None:
        n=len(fileid__list)
    sentences_list = [load_file_sentences(os.path.join(path,fileid))for fileid in fileid_list[:n]]
    reduce
    return reduce(lambda x,y:x+y,sentences_list)

# [debug] load_collection_sentences('/home1/c/cis530/data-hw3/articles/d30006t',2)

# b)
def dependency_parse_sentence(sentence):
    p = Parser()
    deps = p.parseToStanfordDependencies(sentence)
    return [(r,gov.text,dep.text) for r,gov,dep in deps.dependencies]

# sentence="Pick up the tire pallet near the truck."
# Result:[('prt', 'Pick', 'up'), ('det', 'pallet', 'the'), ('nn', 'pallet', 'tire'), ('dobj', 'Pick', 'pallet'), ('det', 'truck', 'the'), ('prep_near', 'pallet', 'truck')]

# c)
def dependency_parse_collection(path):
    dependency_list = [dependency_parse_sentence(s) for s in load_collection_sentences(path,len(os.listdir(path)))]
    return reduce(lambda x,y:x+y, dependency_list)

#dependency_parse_collection('/home1/c/cis530/data-hw3/articles/d30006t')   

# d)
def get_linked_words(dependency_list,word):
    def the_other_word((relation,gov,dep)):
        if gov==word:
            return dep
        elif dep==word:
            return gov
    return list(set(map(the_other_word, dependency_list)))

# dl = dependency_parse_sentence('Pick up the tire pallet near the truck.')
# get_linked_words(dl, 'pallet')


#1.6
# a)
def create_graphviz_file(edge_list,output_file):
    f = open(output_file,'w')
    f.write('graph G {\n')
    for (x,y) in edge_list:
        f.write(x+' -- '+y+';\n')
    f.write('}')
    f.close()    

#[debug] create_graphviz_file([('dog', 'cat'), ('dog', 'computer'), ('cat', 'computer')], 'test.gr')

# b)
def get_top_n_linked_words(topic_word_dict,dependency_list,n,word):
    word_list = get_linked_words(dependency_list,word)
    return filter_top_n_words(topic_word_dict,n,word_list)

#[debug]
#tw = get_topic_words('d30006t.ts')
#dl = dependency_parse_sentence('Pick up the tire pallet near the truck.')
#get_top_n_linked_words(tw, dl, 3, 'pallet')

# c)     
def visualize_collection_topics(topic_file,collection_path,output_file):
    #step1 load a topic file, and get Topic Signature Score (.ts)
    tw = load_topic_words(topic_file)               
    #step2: load collection of document, and get A Sentences List
    cd=load_collection_sentences(collection_path,len(os.listdir(collection_path)))
    #step3: top 10 topic words for cd(Sentences List)      #TODO, SHOULD parse every sentence into word list?                                                
    top_10_tw = get_top_n_topic_words(tw,10)
    #step4: top 5 words linked to each topic word    
    dp = dependency_parse_collection(collection_path)
    #edge_list = [(w,v) for v in get_top_n_linked_words(dps,5,w) for w in top_10_tw]    
    edge_list = [(top_10_tw(i),linked) for linked in get_top_n_linked_words(tw,dp,5,top_10_tw(i)) for i in range(len(top_10_tw))]
    #step5: Create .gr for all links
    create_graphviz_file(edge_list,output_file)

 
    
#[debug]
# topic_file='d30006t.ts'
# collection_path='/home1/c/cis530/data-hw3/articles/d30006t/'   collection_path='d30006t'
# output_file='d30006t.gr' 

#2  Multi-Document Summarization

#2.1 Collecting Features

# a)
def create_feature_space(sentences):             # one functions in HW1
    splits = [s.split() for s in sentences]
    types = set(reduce(lambda x, y: x + y, splits))
    lookup = dict()
    for i, word in enumerate(types):
        lookup[word] = i
    return lookup    

def vectorize(vector_space, sentence):           # another functions in HW1
    vector = [0] * len(vector_space)
    for word in sentence.split():
        vector[vector_space[word]] = 1
    return vector


def create_collection_feature_space(collection_path):       # feature space
    sentence_list = load_collection_sentences(collection_path,len(os.listdir(collection_path)))    
    return create_feature_space(sentence_list)

def vectorize_collection(feature_space,collection_path):    # vectoized collection
    sentence_list = load_collection_sentences(collection_path,len(os.listdir(collection_path)))  
    return [(s,vectorize(feature_space,s)) for s in sentence_list]

#2.2 Ranking Sentences
#2.2.1 Centrality
#a)

def rank_by_centrality(collection_path,sim_func):
    fs = create_collection_feature_space(collection_path)
    vc = vectorize_collection(fs,collection_path)    
    centrality=[(vc[m][0],sum([sim_func(vc[m][1],vc[n][1]) for n in range(len(vc)) if n!=m])) for m in range(len(vc))]   
    return sorted(centrality,key=itemgetter(1),reverse=True)

#[debug] collection_path='/home1/c/cis530/data-hw3/articles/d30006t'
def sim_func(X,Y):      #We use cosine_similarity
    up=sum((X[i]*Y[i]) for i in range(len(X)))
    d1=sum((X[i]*X[i]) for i in range(len(X)))
    d2=sum((Y[i]*Y[i]) for i in range(len(X)))
    return up/(d1*d2)**0.5


#2.2.2 Topic Word Count
def rank_by_tweight(collection_path,topic_file):
    tw = get_topic_words(topic_file)    
    sentence_list = load_collection_sentences(collection_path,len(os.listdir(collection_path)))
    def number_of_tw(sentence):   #number of topic word given a sentence
        return len(set(sentence.split()).intersection(set(tw.keys())))
    return sorted([(s, number_of_tw(s)) for s in sentence_list],key=itemgetter(1),reverse=True)

#[debug]
# topic_file='d30006t.ts'
# collection_path='/home1/c/cis530/data-hw3/articles/d30006t'

#2.3 Performing Summarization

def summarize_ranked_sentences(ranked_sents,summary_len):
    summary = []
    count = 0
    for ranked_tuple in ranked_sents:
        if count+len(ranked_tuple[0].split()) > summary_len:     #ranked_tuple[0] is the ranked sentence
            break
        summary.append(ranked_tuple[0])
        count = count + len(ranked_tuple[0])
    return summary

#[debug] C5=summarize_ranked_sentences(rank_by_centrality('/home1/c/cis530/data-hw3/articles/d31013t', sim_func),150)
#[debug] W5=summarize_ranked_sentences(rank_by_tweight('/home1/c/cis530/data-hw3/articles/d31013t', 'd31013t.ts'),150)
#[debug] d30006t, d30008t, d30037t, d30047t, d31013t





#3.2 Multi-Document Summarization
#3.2 d)
def summarize_ranked_sentences_fixed(ranked_sents,summary_len,min_sentence_length=0,max_sentence_length=9999):
    summary = []
    count = 0
    for ranked_tuple in ranked_sents:
        if min_sentence_length<= len(ranked_tuple[0].split()) <= max_sentence_length:
            if count+len(ranked_tuple[0].split()) > summary_len:     #ranked_tuple[0] is the ranked sentence
                break
            summary.append(ranked_tuple[0])
            count = count + len(ranked_tuple[0])
    return summary

#[debug] C5=summarize_ranked_sentences_fixed(rank_by_centrality('/home1/c/cis530/data-hw3/articles/d31013t', sim_func),150,15,75)
#[debug] W5=summarize_ranked_sentences_fixed(rank_by_tweight('/home1/c/cis530/data-hw3/articles/d31013t', 'd31013t.ts'),150,15,75)


#3.2 e)
# The second fixed summary function. 
def summarize_ranked_sentences_fixed2(ranked_sentences, min_sentence_length=0,max_sentence_length=9999,
                                                                   max_similarity=1, sim_func=sim_func):
    summary = ['']
    count = 0
    for ranked_tuple in ranked_sentences:
        if min_sentence_length<= len(ranked_tuple[0].split()) <= max_sentence_length:
            if count+len(ranked_tuple[0].split()) > 150:    #ignore sentence which makes total > 150
                break
            sentence_list=[s[0] for s in ranked_sentences]
            fs = create_feature_space(sentence_list)         #create a temperate feature space based on current sentences
            summary_vector = vectorize(fs, summary[0])
            new_vector=vectorize(fs, ranked_tuple[0])
            current_similarity=sim_func(new_vector,summary_vector)   # compute similarity using matrix
            if current_similarity>max_similarity:         # ignore sentence which makes current similarity> max_similarity
                break
            summary.append(ranked_tuple[0])
            count = count + len(ranked_tuple[0])
    return summary

#[debug] C5=summarize_ranked_sentences_fixed2(rank_by_centrality('/home1/c/cis530/data-hw3/articles/d31013t', sim_func))
#[debug] W5=summarize_ranked_sentences_fixed2(rank_by_tweight('/home1/c/cis530/data-hw3/articles/d31013t', 'd31013t.ts'))

