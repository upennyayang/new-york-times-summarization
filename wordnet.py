#---------------------------------------------------------------#
#   CIS 530 Homework 4			                        #
#   Yayang Tian	        24963298        yaytian@seas.upenn.edu  #
#   Quan Dong	        18998424        qdong@seas.upenn.edu    #
#---------------------------------------------------------------#

import nltk
import string
import math
import os
from nltk.corpus import wordnet as wn
from operator import itemgetter




#1.1 Polysemous Words

def load_file_words(path):
    input_file = open(path)
    word_list = nltk.word_tokenize(input_file.read().replace('\n','')) #Tokenize it into word list
    input_file.close()
    return word_list

def load_collection_words(path):
    fileid_list=os.listdir(path)    #all the files in the path.
    word_list = [load_file_words(os.path.join(path,fileid))for fileid in fileid_list]
    reduce
    return reduce(lambda x,y:x+y,word_list)


def get_most_polysemous(n,word_list,part_of_speech):
    if part_of_speech=='noun':
        pos=wn.NOUN
    elif part_of_speech=='verb':
        pos=wn.VERB
    elif part_of_speech=='adjective':
        pos=wn.ADJ
    elif part_of_speech=='adverb':
        pos=wn.ADV
    #filter those which haven't such part of speech
    filtered_list=filter(lambda x:wn.synsets(x,pos)!=[],word_list)
    #then sort the set of words according to the number of polysemous words 
    return sorted(set(filtered_list), key=lambda s:len(wn.synsets(s,pos)),reverse=True)[0:n]

def get_least_polysemous(n,word_list,part_of_speech):
    if part_of_speech=='noun':
        pos=wn.NOUN
    elif part_of_speech=='verb':
        pos=wn.VERB
    elif part_of_speech=='adjective':
        pos=wn.ADJ
    elif part_of_speech=='adverb':
        pos=wn.ADV
    #Similarly for least one   
    filtered_list=filter(lambda x:wn.synsets(x,pos)!=[],word_list)
    return sorted(set(filtered_list), key=lambda s:len(wn.synsets(s,pos))>0,reverse=False)[0:n]
            
#[debug]
#word_list = load_collection_words('/home1/c/cis530/data-hw3/articles/d31013t')
#get_most_polysemous(5,word_list,'noun')
#get_least_polysemous(5,word_list,'adjective')


#1.2 Specific Words

#the length to the root of hypernynm of a given word
def shortest_synset_length(word):      
    length_list=[]                              #The lengh to root. They do not contain path whose length=1  
    for synset in wn.synsets(word,wn.NOUN):     #A synset in the synsets list for a word. Concentrate on NOUN       
        paths=synset.hypernym_paths()           #all the paths for a synset
        for path in paths:                      #a path in the paths list
            if (len(path))==1:                  #remove path that has no hypernym
                break
            else:
                length_list.append(len(path))
    if length_list==[]:                         #if there is no hypernynm, set the length zero.
        return -1
    else:
        return min(length_list)                 #Choose the shortest one for multiple paths

def get_most_specific(n, word_list):
    filtered_list=filter(lambda x:shortest_synset_length(x)!=-1,word_list)
    return sorted(set(filtered_list), key=lambda s:shortest_synset_length(s),reverse=True)[0:n]

def get_least_specific(n, word_list):
    filtered_list=filter(lambda x:shortest_synset_length(x)!=-1,word_list)
    return sorted(set(filtered_list), key=lambda s:shortest_synset_length(s),reverse=False)[0:n]
    

#[debug]
#word_list = load_collection_words('/home1/c/cis530/data-hw3/articles/d31013t')
#get_most_specific(5,word_list)
#get_least_specific(5,word_list)



#2 Word Similarity

#a)
def get_similarity(word1, word2):
    if wn.synsets(word1,wn.NOUN)==[] or wn.synsets(word2,wn.NOUN)==[]:
        return -1
    else:
        similarities=[synset1.path_similarity(synset2) for synset1 in wn.synsets(word1, wn.NOUN)
                      for synset2 in wn.synsets(word2, wn.NOUN)]
        return max(similarities)
        
#b)    
def get_all_pairs_similarity(word_list):
    #combination, not permutation, of words and their similarities
    tri=[(word_list[i],word_list[j],get_similarity(word_list[i],word_list[j]))
         for i in range(len(word_list))[0:-1] for j in range(len(word_list))[i+1:len(word_list)]]
    return tri    

            
#c)
def filter_pairs_similarity(pairs, minimum):
    return filter(lambda p:p[2]>=minimum,pairs)

#d)   
def get_similar_groups(word_list, minimum):
    tri_list=get_all_pairs_similarity(word_list)
    tri_filtered=filter_pairs_similarity(tri_list, minimum)
    neighbor=nltk.defaultdict(set)
    for tri in tri_filtered:
        neighbor[tri[0]].add(tri[1])
        neighbor[tri[1]].add(tri[0])

    def bors_kerbosch_v1(R, P, X, G, C): #CODE FROM ONLINE RESOURCE
        if len(P) == 0 and len(X) == 0:
            if len(R) > 2:
                C.append(sorted(R))
            return    
        for v in P.union(set([])):
            bors_kerbosch_v1(R.union(set([v])), P.intersection(G[v]), X.intersection(G[v]), G, C)
            P.remove(v)
            X.add(v)

    def bors_kerbosch_v2(R, P, X, G, C): #CODE FROM ONLINE RESOURCE
        if len(P) == 0 and len(X) == 0:
            if len(R) > 2:
                C.append(sorted(R))
            return
        (d, pivot) = max([(len(G[v]), v) for v in P.union(X)])                  
        for v in P.difference(G[pivot]):
            bors_kerbosch_v2(R.union(set([v])), P.intersection(G[v]), X.intersection(G[v]), G, C)
            P.remove(v)
            X.add(v)
    C = []
    bors_kerbosch_v2(set([]),set(neighbor.keys()),set([]),neighbor,C)
    return C
 
#[debug]
#get_similarity('operation', 'find')
#get_all_pairs_similarity(['settlement', 'camp', 'base', 'country'])
#pairs = get_all_pairs_similarity(['settlement', 'camp', 'base', 'country'])
#filter_pairs_similarity(pairs, 0.25)
#word_list=load_collection_words('/home1/c/cis530/data-hw3/articles/d31013t')



# Writeup
#3.1 

''' First, Run the following command, to convert a ts file into topic word list
<<<topic_file='d31013t.ts'
<<<word_list=[(line.split())[0] for line in open(topic_file).readlines()]

then run
(a)
<<<get_most_polysemous(10,word_list,'noun')
<<<get_most_polysemous(10,word_list,'verb')
<<<get_most_polysemous(10,word_list,'adjective')
<<<get_most_polysemous(10,word_list,'adverb')

(c)
<<<get_least_polysemous(10,word_list,'noun')
<<<get_least_polysemous(10,word_list,'verb')
<<<get_least_polysemous(10,word_list,'adjective')
<<<get_least_polysemous(10,word_list,'adverb')

(d)
<<<get_most_specific(10,word_list)
<<<get_least_specific(10,word_list)

(f)Change the folder from d30006t, d30008t, d30037t, d30047t, to d31013t.
<<<word_list = load_collection_words('/home1/c/cis530/data-hw3/articles/d31013t')
<<<get_most_specific(20,word_list)
<<<get_least_specific(20,word_list)
'''


#3.2

''' First, Run the 5 times of the following command, input d30006t, d30008t, d30037t, d30047t, to d31013t
For example:
<<<topic_file='d31013t.ts'
<<<word_list=[(line.split())[0] for line in open(topic_file).readlines()]

then run
(a)
<<<compute_similar_groups(topic_file, 0.25)
'''







