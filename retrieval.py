# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 23:31:45 2021

@author: Guoyuan An
"""

import numpy as np
from hypergraph_propagation import propagate, prepare_hypergraph_propagation
from utils.retrieval_component import connect_nodup

#retrieval_dataset='roxford'
#retrieval_dataset='rparis'
retrieval_dataset='R1Moxford'
#retrieval_dataset='R1Mparis'

HYPERGRAPH_PROPAGATION=1
#COMMUNITY_SELECTION=0
COMMUNITY_SELECTION=1 # only calculate the uncertainty score, but do not apply geometric verification
#Mention: to use the option COMMUNITY_SELECTION=2, you should extract the local features for all the dataset image firstly.
#COMMUNITY_SELECTION=2 # calculate the uncertainty score. IF the score is large, use geometric verification to find a new start point
 
    
#load the features
if retrieval_dataset=='roxford':
    vecs=np.load('features/roxford_np_delg_features/a_global_vecs.npy').T # (2048,4993)
    qvecs=np.load('features/roxford_np_delg_features/a_global_qvecs.npy').T #(2048,70)

elif retrieval_dataset=='rparis':
    vecs=np.load('features/rparis_np_delg_features/a_global_vecs.npy').T 
    qvecs=np.load('features/rparis_np_delg_features/a_global_qvecs.npy').T 
    
elif retrieval_dataset=='R1Moxford':
    distractors=np.load('features/distractor_np_delg_features/a_global_vecs.npy').T
    vecs=np.load('features/roxford_np_delg_features/a_global_vecs.npy').T # (2048,4993)
    vecs=np.concatenate((vecs,distractors),axis=1) #()
    del distractors
    qvecs=np.load('features/roxford_np_delg_features/a_global_qvecs.npy').T #(2048,70) 
    
elif retrieval_dataset=='R1Mparis':
    distractors=np.load('features/distractor_np_delg_features/a_global_vecs.npy').T
    vecs=np.load('features/rparis_np_delg_features/a_global_vecs.npy').T # (2048,6322)
    vecs=np.concatenate((vecs,distractors),axis=1) #()
    qvecs=np.load('features/rparis_np_delg_features/a_global_qvecs.npy').T #(2048,70)  

qscores=np.dot(vecs.T,qvecs) #(4993,70) 
qranks=np.argsort(-qscores,axis=0) #(4993, 70)

#load the ground truth file
import dataset
from utils.evaluate import compute_map

if retrieval_dataset=='roxford' or retrieval_dataset=='R1Moxford':
    query_list, index_list, ground_truth = dataset.ReadDatasetFile('data/roxford/gnd_roxford5k.mat')
elif retrieval_dataset=='rparis' or retrieval_dataset=='R1Mparis':
    query_list, index_list, ground_truth = dataset.ReadDatasetFile('data/rparis/gnd_rparis6k.mat')

(_, medium_ground_truth,hard_ground_truth) = dataset.ParseEasyMediumHardGroundTruth(ground_truth) # 'ok' and 'junk'

#prepare the hypergraph propagation
if HYPERGRAPH_PROPAGATION==True:
    prepare_hypergraph_propagation(retrieval_dataset)

#prepare the community selection
if COMMUNITY_SELECTION>=True:
    from community_selection import prepare_community_selection,extract_sub_graph,calculate_entropy,match_one_pair_delg,find_dominant
    import community_selection
    
    prepare_community_selection(retrieval_dataset, COMMUNITY_SELECTION)

# start retrieval
total_ranks=[]
for q in range(70):
    print('start query',q )
    
    #first search 
    first_search=list(qranks[:,q])
    dominant_image=first_search[0]
    
    #community selection
    if COMMUNITY_SELECTION>=True:
        #calculate uncertainty
        Gs=extract_sub_graph(first_search,20) #list of set
        uncertainty=calculate_entropy(Gs)
        
        if COMMUNITY_SELECTION==2:
            if uncertainty >1:
                inlier,size=match_one_pair_delg(q, first_search[0])
                
                if inlier<20:
                    #change the dominant image
                    Gs=extract_sub_graph(first_search,100) # top 100
                    dominant_image=find_dominant(Gs,first_search,q)
    
        else:
            print(uncertainty)
            
    #hypergraph propagation
    if HYPERGRAPH_PROPAGATION==True:
        rank_list=propagate([dominant_image])
        #get the final rank
        final_ranks=connect_nodup(rank_list,first_search)
    else:
        final_ranks=first_search
    
    total_ranks.append(final_ranks)
    new_ranks_=np.array(final_ranks).reshape((1,-1))
    medium_metrics = compute_map(new_ranks_.T, [medium_ground_truth[q]]) 
    hard_metrics = compute_map(new_ranks_.T, [hard_ground_truth[q]]) 
    print(medium_metrics[0],hard_metrics[0])
    
total_ranks_=np.array(total_ranks)
medium_metrics = compute_map(total_ranks_.T, medium_ground_truth) 
hard_metrics=compute_map(total_ranks_.T,hard_ground_truth)
print(medium_metrics)
print(hard_metrics)

if COMMUNITY_SELECTION==2:    
    print(community_selection.N_RANSAC)



    
