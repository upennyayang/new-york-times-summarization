#################################
## Yayang Tian (yaytian)       ##
## Fabian Peternek (fape)      ##
## Quan Dong (qdong)           ##
## CIS 530 - Project - Code    ##
#################################

import os
import math
from operator import add

from nltk.tokenize import sent_tokenize

from ordering import compute_similarity_matrix, cluster_sentences, extract_dates,\
        make_themes_from_clusters, chronological_ordering, augmented_ordering,\
        modified_augmented_ordering

__fape_files_to_load = 10


""" The following paths have to be adjusted, for the code below to work. Furthermore, the functions
    assume, that the following directories exist, otherwise there will be an error:
    __project_root/summaries/{CO,AA,mAA}
    __project_root/clusters
    __project_root/topicwords

    To run: First run the "compute_topicwords()" function, which will create topic words files for all
    the collections in __collection_root. After that "make_summaries()" will create 3 summaries for
    every collection. Furthermore "make_summary(collection_name)" and "make_modified_aa_summary(collection_name)"
    will create summaries for a single collection.
"""
__cluto_bin = '/Users/Kirby/Documents/Uni/cis530project/cluto-2.1.2/Darwin-i386/scluster'
__topicwordsbinary = "/Users/Kirby/Documents/Uni/cis530project/TopicWords-v2/TopicSignatures"
__collection_root = "/Users/Kirby/Documents/Uni/cis530project/data/articles/"
__project_root = "/Users/Kirby/Documents/Uni/cis530project/"

#1.2
def load_topic_words(topic_file):
    """ Loads the words from topic file into a dictionary and returns that. """
    f = open(topic_file, 'r')
    # Get word/score combinations into a list
    lines = f.readlines()
    # File is no longer needed
    f.close()
    
    topic_dict = {}
    # Split all the lines, convert to tuples and then put it into the dict
    for (word, score) in map(tuple, map(str.split, lines)):
        topic_dict[word] = float(score)
    return topic_dict

#1.4
def load_file_sentences(filepath, filename):
    """ Loads sentences of a file into a list and converts everything to
    lower case. """
    # Read file as string first
    f = open(filepath, 'r')
    text = f.read()
    f.close()
    # Strip the newlines
    text = filter(lambda x: x != '\n', text)
    # Now use nltks method to read the sentences
    sentences = sent_tokenize(text)
    # convert everything to lower case
    sentences = map(str.lower, sentences)
    """sentences = [(s.lower(), filename) for s in sentences]"""
    # Create segments by clustering. Let's say 3 segments per text.
    # Similarity metric shall be cosine.
    fs = create_feature_space(sentences)
    vectors = [vectorize(fs, sent) for sent in sentences]
    compute_similarity_matrix(vectors, cosine_similarity, filename+".similarities")
    if (len(vectors) < 2):
        # There are not enough sentences to cluster, so we'll just use the same segment for all
        # of them. Only happens once in the given project data anyway.
        segments = [0]*len(vectors)
    else:
        segments = cluster_sentences(filename+".similarities", __cluto_bin, 2)
    # Stitch it all together
    return zip(sentences, [filename]*len(sentences), segments)

def load_collection_sentences(collection, n):
    """ collection is a directory containing text-files. The first n of those
        text files will be loaded as sentences. """
    files = os.listdir(collection)
    files_sentences = []
    for f in files:
        files_sentences.append(load_file_sentences(collection + "/" + f,f))
        n -= 1
        if n == 0:
            break
    return files_sentences

# 2.1
def create_feature_space(sentences):
    """ create feature space from hw1 solution. Creates a feature space. """
    splits = [s.split() for s in sentences]
    types = set(reduce(lambda x, y: x + y, splits))
    lookup = dict()
    for i, word in enumerate(types):
        lookup[word] = i
    return lookup

def vectorize(vector_space, sentence):
    """ vectorize from solution to hw1. Creates a vector for the given sentence
    in accordance to the given feature space. """
    vector = [0] * len(vector_space)
    for word in sentence.split():
        vector[vector_space[word]] = 1
    return vector

def vectorize_sentence_list(feature_space, sentences):
    """ Vectorizes a list of sentences. """
    return map(vectorize, [feature_space]*len(sentences), sentences)

def topic_weight(sentence, topic_words):
    """ Given a sentence and a list of topic words, returns the number of
    topic words in the sentence. """
    topic_words = set(topic_words) # slight speedup, lookup in set is faster
    topic_weight = 0
    for word in sentence.split():
        if word in topic_words:
            topic_weight += 1
    return topic_weight

def rank_by_tweight(collection_path, topic_file):
    """ Ranks the collection in collection_path by topic weight, using the
    topic words found in topic_file. """
    # First get the sentences and the topic words
    ts = load_topic_words(topic_file).keys()
    sentences = load_collection_sentences(collection_path, __fape_files_to_load)
    # reduce to 1-dimensional list of tuples (sentence, filename, segment)
    sentences = reduce(lambda x,y: x+y, sentences)
    # Compute all topic weight values:
    tweights = [(sent, topic_weight(sent[0], ts)) for sent in sentences]
    # Sort descending and return
    tweights.sort(key = lambda x: x[1], reverse=True)
    return tweights

def ordering_preprocessing(collection_name):
    """ Given a collection name, will create the themes for the collection and
        return them, thus enabling to order the themes/sentences.
    """
    print("Starting Preprocessing")
    collection_path = __collection_root + collection_name
    topic_file = __project_root + 'topicwords/' + collection_name + '.ts'
    print("Loading and ranking sentences")
    sentences = rank_by_tweight(collection_path, topic_file)
    sentences = extract_dates(sentences)
    # We need to vectorize all the sentences to cluster them.
    print("Vectorizing Sentences")
    list_of_sents = [sent for ((sent, f, d, s), w) in sentences]
    fs = create_feature_space(list_of_sents)
    vectors = vectorize_sentence_list(fs, list_of_sents)
    # Now we can cluster into themes
    print("Creating Themes")
    compute_similarity_matrix(vectors, cosine_similarity, './similarity_matrix')
    clusters = cluster_sentences('./similarity_matrix', __cluto_bin, 5)
    themes = make_themes_from_clusters(sentences, clusters)
    return themes

# Homework 1 Similarity functions
def intersect(A, B):
    return map(lambda (x, y): min(x, y), zip(A, B))

def sumf(A):
    return float(sum(A))

def cosine_similarity(A, B):
    return sumf(intersect(A, B)) / math.sqrt(sumf(A) * sumf(B))

def make_summary(collection_name):
    """ Creates a single (or in fact two) summary (summaries) of the articles
        in collection_name. Creates a summary using both augmented and chronological
        algorithm and writes them into appropriate files (project_root/summaries/CO/collection_name
        for chronological ordering and project_root/summaries/AA/collection_name for
        augmented algorithm).
    """
    themes = ordering_preprocessing(collection_name)
    COsumm = chronological_ordering(themes)
    AAsumm = augmented_ordering(themes)
    CO_filename = __project_root + 'summaries/CO/' + collection_name
    AA_filename = __project_root + 'summaries/AA/' + collection_name
    fAA = open(AA_filename, 'w')
    fCO = open(CO_filename, 'w')
    fAA.write(reduce(add, AAsumm, ""))
    fCO.write(reduce(add, COsumm, ""))
    fAA.close()
    fCO.close()

def make_modified_aa_summary(collection_name):
    themes = ordering_preprocessing(collection_name)
    topic_file = __project_root + 'topicwords/' + collection_name + '.ts'
    topicwords = load_topic_words(topic_file)
    mAAsumm = modified_augmented_ordering(themes, topicwords)
    mAA_filename = __project_root + 'summaries/mAA/' + collection_name
    fmAA = open(mAA_filename, 'w')
    fmAA.write(reduce(add, mAAsumm, ""))
    fmAA.close()


def make_summaries():
    """ Produces summaries for all the collections. """
    for collection_name in os.listdir(__collection_root):
        #make_summary(collection_name)
        make_modified_aa_summary(collection_name)

## Not ordering, just utility stuff, that's used once.
def compute_topicwords():
    """ Computes all the topicwords-files in one simple function call. """
    for d in os.listdir(__collection_root):
        conf = open(__project_root + 'twords.conf', 'w')
        configstring = "stopFilePath = stoplist-smart-sys.txt\n\
                        performStemming = N\n\
                        backgroundCorpusFreqCounts = bgCounts-Giga.txt\n\
                        topicWordCutoff = 10.0\n"
        configstring += "inputDir = "+ __collection_root + d + "\n"
        configstring += "outputFile = "+ __project_root + "topicwords/"+d+".ts"
        conf.write(configstring)
        conf.close()
        # Run topicwords with that configfile
        #os.system('java -Xmx1000m ' + __topicwordsbinary + ' ' + __project_root + 'twords.conf')
        os.system('java -Xmx1000m TopicSignatures ' + __project_root + 'twords.conf')
