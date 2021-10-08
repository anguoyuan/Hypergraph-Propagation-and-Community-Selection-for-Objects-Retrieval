# Hypergraph Propagation and Community Selection for Objects Retrieval

## How to use the code  
1. Download and unzip the 'graph' and 'features' folders.  
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

