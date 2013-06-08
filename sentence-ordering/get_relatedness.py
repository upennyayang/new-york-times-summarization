#################################
## Yayang Tian (yaytian)       ##
## Fabian Peternek (fape)      ##
## Quan Dong (qdong)           ##
## CIS 530 - Project - Code    ##
## Relatedness measures        ##
#################################

def get_relatedness(theme1,theme2):
    """ Computes relatedness ratio as detailed in the paper for the augmented
        algorithm.
    """
    nAB=0
    nAB_plus=0
    for sentence_1 in theme1.sentences:
        for sentence_2 in theme2.sentences:
            if cmp(sentence_1[2],sentence_2[2])==0:
                nAB=nAB+1
                if sentence_1[3]==sentence_2[3]:
                    nAB_plus=nAB_plus+1
    if nAB==0:
        return 0
    else:
        return float(nAB_plus)/float(nAB)

def topic_weight_relatedness(theme1, theme2, topicword_list):
    """ Computes a new measure of relatedness: Counts how many topic words the
        two themes have alike and how many topic words they have in total. The 
        ratio of these two numbers makes up another measure of relatedness.
    """
    theme1_sents = [sent for (sent, d, f, s, w) in theme1.sentences]
    theme2_sents = [sent for (sent, d, f, s, w) in theme2.sentences]
    t1_words = map(str.split, theme1_sents)
    t2_words = map(str.split, theme2_sents)
    # words are lists of lists now, we want all those in one set though
    t1_words = set(reduce(lambda x,y: x+y, t1_words))
    t2_words = set(reduce(lambda x,y: x+y, t2_words))
    # Make a set out of the topic words, too
    topic = set(topicword_list)
    # Now we can trivialize the whole thing by using set operations, hurray!
    common_topic_words = t1_words.intersection(t2_words).intersection(topic)
    total_topic_words = (t1_words.union(t2_words)).intersection(topic)
    return float(len(common_topic_words))/float(len(total_topic_words))
