from tqdm import tqdm
import numpy as np
import os.path, pickle
from utils import obtain_all_seed_concepts
from utils_graph import conceptnet_graph, domain_aggregated_graph, subgraph_for_concept

if __name__ == '__main__':
    
    bow_size = 5000
    
    print ('Extracting seed concepts from all domains.')
    all_seeds = obtain_all_seed_concepts(bow_size)
    
    print ('Creating conceptnet graph.')
    G, G_reverse, concept_map, relation_map = conceptnet_graph('conceptnet_english.txt')
    
    print ('Num seed concepts:', len(all_seeds))
    print ('Populating domain aggregated sub-graph with seed concept sub-graphs.')
    triplets, unique_nodes_mapping = domain_aggregated_graph(all_seeds, G, G_reverse, concept_map, relation_map)
    
    print ('Creating sub-graph for seed concepts.')
    concept_graphs = {}

    for node in tqdm(all_seeds, desc='Instance', position=0):
        concept_graphs[node] = subgraph_for_concept(node, G, G_reverse, concept_map, relation_map)
        
    # Create mappings
    inv_concept_map = {v: k for k, v in concept_map.items()}
    inv_unique_nodes_mapping = {v: k for k, v in unique_nodes_mapping.items()}
    inv_word_index = {}
    for item in inv_unique_nodes_mapping:
        inv_word_index[item] = inv_concept_map[inv_unique_nodes_mapping[item]]
    word_index = {v: k for k, v in inv_word_index.items()}
        
    print ('Saving files.')
        
    pickle.dump(all_seeds, open('utils/all_seeds.pkl', 'wb'))
    pickle.dump(concept_map, open('utils/concept_map.pkl', 'wb'))
    pickle.dump(relation_map, open('utils/relation_map.pkl', 'wb'))
    pickle.dump(unique_nodes_mapping, open('utils/unique_nodes_mapping.pkl', 'wb'))
    pickle.dump(word_index, open('utils/word_index.pkl', 'wb'))
    pickle.dump(concept_graphs, open('utils/concept_graphs.pkl', 'wb'))
    
    np.ndarray.dump(triplets, open('utils/triplets.np', 'wb'))        
    print ('Completed.')