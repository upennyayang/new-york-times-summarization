#################################
## Yayang Tian (yaytian)       ##
## Fabian Peternek (fape)      ##
## Quan Dong (qdong)           ##
## CIS 530 - Project - Code    ##
## Ordering module             ##
#################################

from datetime import date
from theme_block import Theme, Block
from get_relatedness import get_relatedness, topic_weight_relatedness

import os

def extract_dates(sentence_list):
    """ Given a set of ranked sentences and their filenames extracts the dates
        from the filenames.
        sentence_list is expected to have the format 
            ((sentence, filename), weight)
        and the filenames look like 'ABCYYYYMMDD.xxxx.clean'.
        This function returns the sentences in the same order as it receives but
        includes a date object in additon to the filename.
    """
    output_list = []
    for ((sent, f_name, segment), w) in sentence_list:
        date_string = f_name.split('.')[0][3:]
        year = int(date_string[:4])
        month = int(date_string[4:6])
        day = int(date_string[6:8])
        output_list.append(((sent, date(year,month,day), f_name, segment), w))
    return output_list

def chronological_ordering(themes):
    """ Applies the chronological ordering algorithm on the given set of themes, 
        This function basically returns the finished summary as it returns just
        the sentences representing the themes.
    """
    # Sort by date of publishment
    print("Chronological ordering")
    themes.sort()
    # return the summary sentences
    return [thm.most_informative[0] + " " for thm in themes]

def augmented_ordering(themes):
    """ Applies the augmented algorithm on the given themes, so that
        the finished summary is returned as it returns just the most informative
        sentence for every theme.
    """
    print("Augmented ordering")
    # First compute ratio of relatedness graph for the themes
    graph = create_relatedness_graph(themes)
    # compute transitive closure of the graph
    graph = transitive_closure(graph)
    # Get the connected components which we'll use as the blocks
    components = compute_components(graph)
    blocks = []
    for component in components:
        block = Block()
        for theme in component:
            block.add_theme(theme)
        blocks.append(block)
    # order the blocks chronologically ascending.
    blocks.sort()
    # order the themes in every block chronologically ascending
    for block in blocks:
        block.themes.sort()
    # Finally get the most informative sentences
    summary = []
    for block in blocks:
        for theme in block.themes:
            summary.append(theme.most_informative[0] + " ")
        summary.append('\n\n') # This way we can see in the summary where blocks ended
    return summary

def modified_augmented_ordering(themes, topicword_list):
    """ Applies the augmented algorithm on the given themes, so that
        the finished summary is returned as it returns just the most informative
        sentence for every theme.
    """
    print("Modified Augmented ordering")
    # First compute ratio of relatedness graph for the themes
    graph = create_tweight_relatedness_graph(themes, topicword_list)
    # compute transitive closure of the graph
    graph = transitive_closure(graph)
    # Get the connected components which we'll use as the blocks
    components = compute_components(graph)
    blocks = []
    for component in components:
        block = Block()
        for theme in component:
            block.add_theme(theme)
        blocks.append(block)
    # order the blocks chronologically ascending.
    blocks.sort()
    # order the themes in every block chronologically ascending
    for block in blocks:
        block.themes.sort()
    # Finally get the most informative sentences
    summary = []
    for block in blocks:
        for theme in block.themes:
            summary.append(theme.most_informative[0] + " ")
        summary.append('\n\n') # This way we can see in the summary where blocks ended
    return summary

def transitive_closure(graph):
    """ Computes the transitive closure of a given graph. 
        The Algorithm is straight-forward and fairly inefficient, but
        as our graphs shouldn't get very large, that probably won't matter.
        Just compute all possible paths from every node and add the new edges.
        graph is given as tuple of two lists: One containing the vertices and
        one containing the edges.
        Algorithm first computes the connected components of the graph by BFS
        and then basically just makes cliques out of every component.
        Complexity should be somewhere around O(VE+V^2).
    """
    # Compute components and put use their carthesian product as edges
    components = compute_components(graph)
    V = set(graph[0])
    E = set()
    for comp in components:
        E = E.union(set([frozenset([u,v]) for u in comp for v in comp\
                if id(u) != id(v)]))
    return (list(V), [tuple(e) for e in E])

def exists_path(start, end, (V,E)):
    """ Given start and end vertex, checks if there is a path from start to end
        in the given Graph. 
        Algorithm uses BFS.
    """
    return end in get_reachable_vertices(start, (V,E)) if start != end\
            else False

def compute_components(graph):
    """ Computes the connected components of the given graph and returns them as
        a list of vertexsets.
    """
    # Setify the graph
    V = set(graph[0])
    E = set([frozenset(edge) for edge in graph[1]])
    components = []
    # While unprocessed vertices exist
    while len(V) != 0:
        # Take any vertex and compute the connected component it belongs to
        v = iter(V).next()
        reachable = get_reachable_vertices(v, (V,E))
        components.append(reachable)
        V = V.difference(reachable)
    return components

def get_reachable_vertices(node, (V,E)):
    """ Implements BFS to find connected component. """
    to_visit = [u for u in V if frozenset([node,u]) in E]
    seen = set([node])
    while len(to_visit) != 0:
        v = to_visit.pop()
        if v not in seen:
            # get new neighborhood and add v to the component
            to_visit = [u for u in V if frozenset([v,u]) in E] + to_visit
            seen.add(v)
    return seen

def compute_similarity_matrix(vectors, sim_func, out_file):
    """ Computes pairwise similarities of all sentences represented by the
        featurespace using sim_func as similarity metric. Writes a matrix that 
        can be used as input for Cluto clustering into out_file.
    """
    outstring = str(len(vectors)) + "\n"
    for s1 in vectors:
        for s2 in vectors:
            outstring += str(sim_func(s1,s2)) + " "
        outstring += "\n"
    # Similarity matrix computed, write to file:
    f = open(out_file, 'w')
    f.write(outstring)
    f.close()

def cluster_sentences(similarity_matrix_file, cluto_bin, num_clusters=5):
    """ Uses Cluto to produce a clustering of the given similarity matrix.
        Returns the clustering vector.
    """
    clustfile = "./clusters/"+similarity_matrix_file+"."+str(num_clusters)
    os.system(cluto_bin + " -clustfile=" + clustfile + " " + similarity_matrix_file + " " + str(num_clusters))
    # now get the clusters
    f = open(clustfile, 'r')
    clusters = f.readlines()
    f.close()
    return clusters

def make_themes_from_clusters(sentences, clusters):
    """ Given a set of sentences and the clusters they belong to
        constructs a list of themes.
        Sentences is a list of tuples, that have the following form:
        ((sentence, date, filename), topic_weight)
    """
    # First create empty theme for every cluster
    themes = [Theme() for i in set(clusters)]
    # Now add every sentence into the cluster/theme it belongs to
    for (i, sent) in enumerate(sentences):
        themes[int(clusters[i])].add_sentence(sent)
    return themes

def create_relatedness_graph(themes):
    """ Computes pairwise relatedness of the themes and returns a
        graph, that has the themes as nodes and an edge between two
        nodes, if the themes' relatedness is >=0.6
    """
    V = themes
    E = []
    for t1 in themes:
        for t2 in themes:
            if id(t1) != id(t2): # We don't want reflexive edges
                if get_relatedness(t1,t2) >= 0.6:
                    E.append((t1,t2))
    return (V,E)

def create_tweight_relatedness_graph(themes, topicwords):
    """ Computes pairwise relatedness of the themes and returns a
        graph, that has the themes as nodes and an edge between two
        nodes, if the themes' relatedness is >=0.6
    """
    V = themes
    E = []
    for t1 in themes:
        for t2 in themes:
            if id(t1) != id(t2): # We don't want reflexive edges
                if topic_weight_relatedness(t1,t2,topicwords) >= 0.5:
                    E.append((t1,t2))
    return (V,E)
