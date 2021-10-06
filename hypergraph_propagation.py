# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 23:32:51 2021

@author: Guoyuan An
"""


import numpy as np
from utils.retrieval_component import connect_nodup

def prepare_hypergraph_propagation(dataset):   
    graph_directaries={'roxford':'graph/delg/roxford/0301/',
                 'rparis': 'graph/delg/rparis/',
                 'R1Moxford': 'graph/delg/R1Moxford/',
                 'R1Mparis':'graph/delg/R1Mparis/'}
    
    global graph_dir, Neighbors, Match_Region, Match_Loc
    
    try:
        graph_dir=graph_directaries[dataset]
    except:
        print('only allow rparis, roxford, R1Moxford, and R1Mparis')
    
    Neighbors=np.load(graph_dir+'Neighbors.npy',allow_pickle=True).item() #dict. key is image index, value is list of neighbor image indexes
    Match_Region=np.load(graph_dir+'Inlier_Region.npy',allow_pickle=True).item() #dict, key is like 'imlist_2178_imlist_4992' 
    Match_Loc=np.load(graph_dir+'Inlier_Loc.npy',allow_pickle=True).item() #key is like "qimlist_26_imlist_696", value is the numpy array with shape(9,2)

   

def propagate(start_list):
    #update Up_Stop_Region for start_list
    global Up_Stop_Region
    Up_Stop_Region={} #key is img index, value is like (up,down,left,right)
    for i in start_list:
        Up_Stop_Region[i]=(0,200000,0,20000)
    
    new_cases=start_list[:]
    rank_list=new_cases[:]
    for _ in range(3):
        pair_list=_expander(new_cases) # return list of ['imlist_2178_imlist_4992',... ]
        new_cases=_adopter(pair_list)
        #new_cases=adopt_all(pair_list) # return list of image number
        new_cases=_no_repeat(new_cases)
        rank_list=connect_nodup(rank_list,new_cases)
        #len(pair_list),len(new_cases),len(rank_list)
        if len(new_cases)==0:
            break
    return rank_list





def _no_repeat(origin_list):
    # remove the repeat ones and 
    list_set=set(origin_list)
    unique_number=len(list_set)
    
    new_list=[]
    for x in origin_list:
        if x in list_set:
            new_list.append(x)
            list_set.remove(x)
        if len(new_list)==unique_number:
            break
    return new_list
        

    
def _expander(up_imgs):
    '''
    Find the neighbors in next hop, return the list of pairs

    Parameters
    ----------
    up_imgs : list of image index
        DESCRIPTION.

    Returns
    -------
    pair_list : list of all the pairs, item is like 'imlist_2178_imlist_4992' 
        DESCRIPTION.

    '''
    pair_list=[]
    for img in up_imgs:
        try:
            pairs=['imlist_'+str(img)+'_imlist_'+str(x) for x in Neighbors[img]]
            pair_list=pair_list+pairs
        except:
            print(img)
        
    return pair_list

def _adopter(pair_list):
    adopted_cases=[]
    for pair in pair_list:
        A,B=int(pair.split('_')[1]),int(pair.split('_')[3])
        reverse_key='imlist_'+str(B)+'_imlist_'+str(A)
        if reverse_key==pair:
            continue
        
        #find the down_stop_region and down_stop_inliers of the up_img
        A_up_stop_region=Up_Stop_Region[A]
        A_emit_region=Match_Region[reverse_key]
        A_emit_inlier_locs=Match_Loc[reverse_key]
        A_down_stop_region,A_down_stop_inliers=_find_down_stop(A_up_stop_region,A_emit_region,A_emit_inlier_locs)
        
        if len(A_down_stop_inliers)==0:
            continue
        else:
            #find the up_stop_inlier and up_stop_region of the down_img
            B_accept_inlier_locs=Match_Loc[pair]
            B_up_stop_inlier_locs=B_accept_inlier_locs[A_down_stop_inliers,:]
            if B not in Up_Stop_Region.keys():    
                Up_Stop_Region[B]=_inlier_Region(B_up_stop_inlier_locs)
                adopted_cases.append(B)
            else:
                Up_Stop_Region[B]=_region_union(Up_Stop_Region[B],_inlier_Region(B_up_stop_inlier_locs))
            
    return adopted_cases



def _find_down_stop(up_stop_region, emit_region,emit_inlier_locs):
    '''
    find the down_stop_region and down_stop_inlier in the down_img

    Parameters
    ----------
    up_stop_region : (up,down,left,right)
        DESCRIPTION.
    emit_region : (up,down,left,right)
        DESCRIPTION.
    emit_inlier_locs : the emit inlier locations of the up_img
        DESCRIPTION.

    Returns
    -------
    down_stop_region : (up,down,left,right)
        DESCRIPTION.
    down_stop_inlier : list of inlier index in the emit_inlier (up_img) and accept_inlier (down_img)
        DESCRIPTION.

    '''
    
    #find down_stop_region 
    down_stop_region=_region_intersection(up_stop_region, emit_region)
    
    #find down_stop_inlier
    down_stop_inlier=[]
    for i,loc in enumerate(emit_inlier_locs):
        if loc[0]>down_stop_region[0] and loc[0]<down_stop_region[1]  and loc[1]>down_stop_region[2]  and loc[1]<down_stop_region[3]:
            down_stop_inlier.append(i)
    
    return down_stop_region,down_stop_inlier
        
def _inlier_Region(inlier_locs):
    
    down,up=np.max(inlier_locs[:,0]),np.min(inlier_locs[:,0])
    right,left=np.max(inlier_locs[:,1]),np.min(inlier_locs[:,1])
    
    return (up,down,left,right)        
        
def _region_intersection(first,second):
    '''
    return the intersection of two region

    Parameters
    ----------
    first : (up,down, left,right)
        DESCRIPTION.
    second : (up,down, left,right)
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
    up=max(first[0],second[0])
    down=min(first[1],second[1])
    left=max(first[2],second[2])
    right=min(first[3],second[3])    
    
    return (up,down, left,right)
    
def _region_union(first, second):
    '''
    return the union of two regions

    Parameters
    ----------
    first : (up,down, left,right)
        DESCRIPTION.
    second : (up,down, left,right)
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
    up=min(first[0],second[0])
    down=max(first[1],second[1])
    left=min(first[2],second[2])
    right=max(first[3],second[3])    
    
    return (up,down, left,right)
        

