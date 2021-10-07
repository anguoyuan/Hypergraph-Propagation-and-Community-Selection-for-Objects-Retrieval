# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:02:42 2021

@author: Guoyuan An
"""

#prepare the uncertainty calculation
from scipy.stats import entropy
import numpy as np
import pickle

def prepare_community_selection(dataset,COMMUNITY_SELECTION):  
    
    graph_directaries={'roxford':'graph/delg/roxford/0301/',
                 'rparis': 'graph/delg/rparis/',
                 'R1Moxford': 'graph/delg/R1Moxford/',
                 'R1Mparis':'graph/delg/R1Mparis/'}
    
    global _graph_dir, Neighbors,_dataset_meta, _dataset
    _dataset=dataset
    
    try:
        graph_dir=graph_directaries[_dataset]
    except:
        print('only allow rparis, roxford, R1Moxford, and R1Mparis')
    
    #load Neighbors
    Neighbors=np.load(graph_dir+'Neighbors.npy',allow_pickle=True).item() #dict. key is image index, value is list of neighbor image indexes
   
    #load the ground truth file
    if _dataset=='roxford':
        with open('data/roxford/gnd_roxford5k.pkl','rb') as f:
            _dataset_meta=pickle.load(f)
    elif _dataset=='rparis':
        with open('data/rparis/gnd_rparis6k.pkl','rb') as f:
            _dataset_meta=pickle.load(f)
       
    if COMMUNITY_SELECTION==2:
        #prepare the delg match 
        global _geom,_features,_qgeom,_qfeatures
        
        def _read_delg_index():
            #return the list of index features and locations; list of array
            geom,features=[],[]
            for img in _dataset_meta['imlist']:
                geom_path='features/'+_dataset+'_np_delg_features/'+img+'_local_locations.npy'
                features_path='features/'+_dataset+'_np_delg_features/'+img+'_local_descriptors.npy'
                geom.append(np.load(geom_path))
                features.append(np.load(features_path))
                
            return geom, features
        
        _geom,_features=_read_delg_index()
        
        def _read_delg_query():
            #return the list of index features and locations; list of array
            geom,features=[],[]
            for img in _dataset_meta['qimlist']:
                geom_path='features/'+_dataset+'_np_delg_features/'+img+'_local_locations.npy'
                features_path='features/'+_dataset+'_np_delg_features/'+img+'_local_descriptors.npy'
                geom.append(np.load(geom_path))
                features.append(np.load(features_path))
                
            return geom, features
        
        _qgeom,_qfeatures=_read_delg_query()

missing=[]
def extract_sub_graph(first_search,bound=100):
    def add_to_subgraph(img):
        potential=[]
        # If added to existing sub_graph, return True
        for i,s in enumerate(sub_graph):
            if len( set(Neighbors[img]).intersection(s) )!=0:
                potential.append(s)
                sub_graph.remove(s)
        
        if len(potential)==1:
            s=potential[0]
            s.add(img)
            sub_graph.append(s)
            return True
        
        elif len(potential)>1:
            s=set([img])
            for x in potential:
                s.update(x)
            sub_graph.append(s)
            return True
        
        else:
            return False
                
    sub_graph=list() #list of set, 由很多的connected components组成
    for i,img in enumerate(first_search[:bound]):
        
        #try to add to existing sub_graph
        tmp=add_to_subgraph(img) #tmp is True if succesfully add
        
        #otherwise, try to connect with other remaining nodes
        if tmp==False:
            s=set(Neighbors[img]).intersection(first_search[i:bound])
            s.add(img) # in case Neighbors[img] doesn't contain img
            sub_graph.append(s)
            
    return sub_graph

def calculate_entropy(sub_graph):
    #calculate entropy
    
    length=sum([len(s) for s in sub_graph])
    numbers=[len(community)/length for community in sub_graph]
    e=entropy(numbers)
    return e



def _inlier_Region(inlier_locs):
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
    
    size=(right-left)*(down-up)
    
    return size  

N_RANSAC=0

from utils.image_reranking import MatchFeatures
def match_one_pair_delg(query, index):
    global N_RANSAC
    N_RANSAC+=1
    
    query_locations=_qgeom[query]
    query_descriptors=_qfeatures[query]
    index_locations=_geom[index]
    index_descriptors=_features[index]
        
    # inlier_locations: list [array([400., 176.]),...]
    inlier_number,match_viz_io,query_inlier_locations,index_inlier_locations,q2i,i2q=MatchFeatures(
        query_locations,query_descriptors,
        index_locations, index_descriptors, 
        ransac_seed=None, descriptor_matching_threshold=0.9,
        ransac_residual_threshold=10.0,use_ratio_test=False)
    
    if inlier_number != 0:
        query_inlier_locations=np.array(query_inlier_locations)
        index_inlier_locations=np.array(index_inlier_locations) # np array 例(6,2)
        
        index_size=_inlier_Region(index_inlier_locations)
        query_size=_inlier_Region(query_inlier_locations)
    
        return inlier_number,index_size
    else:
        return 0,999999999
################################################################################


#to find the new dominant image
import cv2
    
def find_dominant(Gs,first_search,q):
    
    #use sift for spatial matching
    dominant_image=first_search[0]
    for i in first_search[:100]:
        global N_RANSAC
        N_RANSAC+=1
        
        img1 = cv2.imread('data/oxford5k/jpg/'+_dataset_meta['qimlist'][q]+'.jpg',cv2.IMREAD_COLOR)  
        a,b,c,d=_dataset_meta['gnd'][q]['bbx']
        left, upper, right, lower=int(a),int(b),int(c),int(d) #(left, upper, right, lower)
        img1=img1[upper:lower,left:right] #[y1:y2, x1:x2]
        gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        
        img2 = cv2.imread('data/oxford5k/jpg/'+_dataset_meta['imlist'][i]+'.jpg',cv2.IMREAD_COLOR)  
        gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        
        
        sift = cv2.SIFT_create()
        #  使用SIFT查找关键点key points和描述符descriptors
        kp1,des1 = sift.detectAndCompute(gray1, None)
        kp2,des2 = sift.detectAndCompute(gray2, None)
        
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1,des2,k=2)
     
        # store all the good matches as per Lowe's ratio test.
        # we don't allow a feature in index image to be matched by several query features
        good = []
        matched_features=set()
        for m,n in matches:    
            if m.distance < 0.7*n.distance and (m.trainIdx not in matched_features):
                good.append(m)
                matched_features.add(m.trainIdx)
        
        if len(good)<4:
            continue
        
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        
        if sum(matchesMask)>20:
            dominant_image=i
            break   
    return dominant_image