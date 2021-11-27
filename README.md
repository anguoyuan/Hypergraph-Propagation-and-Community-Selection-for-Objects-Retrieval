# Hypergraph Propagation and Community Selection for Objects Retrieval
This is the code of our NeurIPS2021 paper: Hypergraph Propagation and Community Selection for Objects Retrieval. You can check the detail from our project page: https://sgvr.kaist.ac.kr/~guoyuan/hypergraph_propagation/ . 

## to use the code  
1. Download and unzip the 'features' and 'graph' folders. They are the global DELG features and our precomputed matching information. If you want to try on different features, you can get the matching information by running the offline computing code.
2. Open retrieval.py, and choose the retrieval dataset and whether to use hypergraph propagation and community selection.  
3. run retrieval.py  
4. To implement geometric verificatoin, you need to download the ROxford/RParis datasets and extract their local features, which can be achieved by using Radenovic and Cao's open code: https://github.com/filipradenovic/revisitop.git and https://github.com/tensorflow/models/tree/master/research/delf  

The running directory structure should be:

├─data  
│  ├─roxford  
│  └─rparis  
├─features  
│  ├─distractor_np_delg_features   
│  ├─roxford_np_delg_features   
│  └─rparis_np_delg_features  
├─graph  
│  └─delg  
│ | | | ├─R1Moxford  
│ | | | ├─R1Mparis  
│ | | | ├─roxford  
│ | | | └─rparis  
├─utils  

## offline pre-computing  
under Hypergraph_Propagation_and_Community_Selection_for_Objects_Retrieval/  
python  
from offline_RANSAC import pre_match   
pre_match(0,100000,2)  

Ack: The main matching code and parameter setting is copy from DELG open code.  
We only have the cpu version currently. We used several cpus and a large amount of memory to get the result on R1M distractors. 
However, the complexity of the offline process is O(1). And gpu will accelarate this process a lot theratically.  
We would be grateful if anyone offers a (or introduce an existing) gpu version geometric-verification code.
