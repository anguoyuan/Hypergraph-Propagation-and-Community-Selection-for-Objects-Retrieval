# -*- coding: utf-8 -*-
"""
This is the offline RANSAC calculation for hypergraph propagation.

To use: 
under Hypergraph_Propagation_and_Community_Selection_for_Objects_Retrieval/
import pre_match 
pre_match(0,100000,2)

This is the cpu version. 
To open multi threads for accelarating dataset with R1M distractors, several cpus and a large amount of memory are needed. 
You may want to change the (#get the pre matching pairs) and (#load the local descriptors) code based on your equipment.

We would be grateful if anyone offers a gpu version geometric-verification code.
                  
Created on Sat Mar 13 09:57:28 2021

@author: Guoyuan An
"""

import numpy as np
import pickle
from utils.image_reranking import MatchFeatures

dataset='rparis6k'
#dataset='roxford5k'

with open('data/'+dataset[:-2]+'/gnd_'+dataset+'.pkl','rb') as f:
    roxford=pickle.load(f)
    
#oxford 的features
vecs=np.load('features/'+dataset[:-2]+'_np_delg_features/a_global_vecs.npy').T #(2048,6322)
qvecs=np.load('features/'+dataset[:-2]+'_np_delg_features/a_global_qvecs.npy').T #(2048,70)

qscores=np.dot(vecs.T,qvecs) #(4993,70)
qranks=np.argsort(-qscores,axis=0) #(4993, 70)

qscore_ranks=np.sort(-qscores,axis=0) #(4993,70)

scores=np.dot(vecs.T,vecs) #(4993,4993)
ranks=np.argsort(-scores,axis=0) #(4993,4993)

graph_dir='graph/delg/'+dataset[:-2]+'/' #'graph/delg/rparis/'

#get the pre matching pairs
def sp_pairs():
    
    all_pairs_200=set() # 记录所有可能需要计算ransac的pairs，既每张图片与其global的前200张
    for i in range(vecs.shape[1]):
        top200=list(ranks[:200,i])
        for x in top200:
            if 'imlist_'+str(x)+'_imlist_'+str(i) not in all_pairs_200:
                all_pairs_200.add('imlist_'+str(i)+'_imlist_'+str(x))
                
    all_pairs_200=list(all_pairs_200)
    np.save(graph_dir+'pre_pairs.npy',all_pairs_200)

sp_pairs()
all_pairs_200=np.load(graph_dir+'pre_pairs.npy',allow_pickle=True)


#load the local descriptors
def read_delg_index():
    #return the list of index features and locations; list of array
    geom,features=[],[]
    for img in roxford['imlist']:
        geom_path='features/'+dataset[1:]+'_np_delg_features/'+img+'_local_locations.npy'
        features_path='features/'+dataset[1:]+'_np_delg_features/'+img+'_local_descriptors.npy'
        geom.append(np.load(geom_path))
        features.append(np.load(features_path))
        
    return geom, features

geom,features=read_delg_index()
    

#initialize the dictionaries
def initialize_dict():
    '''
    initialize the dictionaries
    To continue from former breakpoint

    '''

    try:
        Neighbors=np.load(graph_dir+'Neighbors.npy',allow_pickle=True).item()
    except:
        Neighbors={} #key is im_list index, value is the list of neighbors 
    
    try:
        Inlier_Num=np.load(graph_dir+'Inlier_Num.npy',allow_pickle=True).item()
    except:
        Inlier_Num={} #key is like "qimlist_26_imlist_696", value is the inlier number 
    
    try:
        Inlier_Loc=np.load(graph_dir+'Inlier_Loc.npy',allow_pickle=True).item()
    except:
        Inlier_Loc={} #key is like "qimlist_26_imlist_696", value is the list of inlier locations on the index image
    
    try:
        Inlier_Index=np.load(graph_dir+'Inlier_Index.npy',allow_pickle=True).item()
    except:
        Inlier_Index={} #key is like "qimlist_26_imlist_696", value is np array like [(q_feature, i_feature),...]
    
    try:
        Inlier_Region=np.load(graph_dir+'Inlier_Region.npy',allow_pickle=True).item()
    except:
        Inlier_Region={} #key is like "qimlist_26_imlist_696", value is the list of inlier region on the index image

    return Neighbors, Inlier_Num, Inlier_Loc, Inlier_Index, Inlier_Region

_, Inlier_Num, Inlier_Loc, Inlier_Index, Inlier_Region=initialize_dict()




#offline match
def pre_match(s,t,thread_id):
    
    pairs=all_pairs_200[s:t]
    times=s
    for pair in list(pairs):
        #calculate and save the inlier number, locations, indexes and regions 
        match_one_pair(pair)
        
        times+=1
        if times%200 == 0:
            print('finished', times)
            np.save(graph_dir+'Inlier_Num'+str(thread_id)+'.npy',Inlier_Num)
            np.save(graph_dir+'Inlier_Loc'+str(thread_id)+'.npy',Inlier_Loc)
            np.save(graph_dir+'Inlier_Index'+str(thread_id)+'.npy',Inlier_Index)
            np.save(graph_dir+'Inlier_Region'+str(thread_id)+'.npy',Inlier_Region)
            
    print('finished', times)      
    np.save(graph_dir+'Inlier_Num'+str(thread_id)+'.npy',Inlier_Num)
    np.save(graph_dir+'Inlier_Loc'+str(thread_id)+'.npy',Inlier_Loc)
    np.save(graph_dir+'Inlier_Index'+str(thread_id)+'.npy',Inlier_Index)
    np.save(graph_dir+'Inlier_Region'+str(thread_id)+'.npy',Inlier_Region)
        
        
def match_one_pair(pair):
    query,index=int(pair.split('_')[1]),int(pair.split('_')[3])
    if pair not in Inlier_Index.keys():
        query_locations=geom[query]
        query_descriptors=features[query]
        index_locations=geom[index]
        index_descriptors=features[index]
        
        # inlier_locations: list [array([400., 176.]),...]
        inlier_number,_,query_inlier_locations,index_inlier_locations,q2i,i2q=MatchFeatures(
            query_locations,query_descriptors,
            index_locations, index_descriptors, 
            ransac_seed=None, descriptor_matching_threshold=0.9,
            ransac_residual_threshold=10.0,use_ratio_test=False)
        
        Inlier_Num['imlist_'+str(query)+'_imlist_'+str(index)]=inlier_number
        Inlier_Num['imlist_'+str(index)+'_imlist_'+str(query)]=inlier_number
        
        if inlier_number != 0:
            query_inlier_locations=np.array(query_inlier_locations)
            index_inlier_locations=np.array(index_inlier_locations) # np array 例(6,2)
            
            index_region=inlier_Region(index_inlier_locations)
            query_region=inlier_Region(query_inlier_locations)
            
            
            
            Inlier_Loc['imlist_'+str(query)+'_imlist_'+str(index)]=index_inlier_locations
            Inlier_Loc['imlist_'+str(index)+'_imlist_'+str(query)]=query_inlier_locations
            Inlier_Index['imlist_'+str(query)+'_imlist_'+str(index)]=q2i
            Inlier_Index['imlist_'+str(index)+'_imlist_'+str(query)]=i2q
            Inlier_Region['imlist_'+str(query)+'_imlist_'+str(index)]=index_region
            Inlier_Region['imlist_'+str(index)+'_imlist_'+str(query)]=query_region
        
    
def inlier_Region(inlier_locs):
    '''
    calculate the inlier region

    Parameters
    ----------
    inlier_locs : numpy array (72,2)
        DESCRIPTION.

    Returns
    -------
    up : TYPE
        DESCRIPTION.
    down : TYPE
        DESCRIPTION.
    left : TYPE
        DESCRIPTION.
    right : TYPE
        DESCRIPTION.

    '''
    
    down,up=np.max(inlier_locs[:,0]),np.min(inlier_locs[:,0])
    right,left=np.max(inlier_locs[:,1]),np.min(inlier_locs[:,1])
    
    return (up,down,left,right)   



def merge_dictions(thread_ids):
    #merge the result of different threads
    
    Number_set,Location_set, Index_set, Region_set={},{},{},{}
    for i in thread_ids:
        Num=np.load(graph_dir+'Inlier_Num'+str(i)+'.npy',allow_pickle=True).item()
        Loc=np.load(graph_dir+'Inlier_Loc'+str(i)+'.npy',allow_pickle=True).item()
        Index=np.load(graph_dir+'Inlier_Index'+str(i)+'.npy',allow_pickle=True).item()
        Region=np.load(graph_dir+'Inlier_Region'+str(i)+'.npy',allow_pickle=True).item()
        
        Number_set.update(Num)
        Location_set.update(Loc)
        Index_set.update(Index)
        Region_set.update(Region)
        
    np.save(graph_dir+'Inlier_Num.npy',Number_set)
    np.save(graph_dir+'Inlier_Loc.npy',Location_set)
    np.save(graph_dir+'Inlier_Index.npy',Index_set)
    np.save(graph_dir+'Inlier_Region.npy',Region_set)  

            
            

def check_missing():
    #check if there is missing pairs
    all_pairs_200=np.load(graph_dir+'pre_pairs.npy',allow_pickle=True)
    current=set(Inlier_Num.keys())
    
    missing=[]
    for i in all_pairs_200:
        if i not in current:
            missing.append(i)
            
    np.save('missing_pairs.npy',missing)
 

    
    
    
    
    
    
    
    
        