# Diffusion/propagation can boost retrieval performance without any additional training.
* If you only have a global feature for each image and want a convenient performance improvement, try ordinary graph diffusion ([Tutorial](https://github.com/anguoyuan/Diffusion-for-retrievl-python), Dmytro's [implementation](https://github.com/ducha-aiki/manifold-diffusion))

* If you want a better performance, try the following hypergraph propagation solution. To the best of our knowledge, this is the current (2023/06/30) best open-source result on ROxford and RParis. The additional memory and time cost is low compared with other reranking approaches, such as spatial verification.

 ![image](https://github.com/anguoyuan/Hypergraph-Propagation-and-Community-Selection-for-Objects-Retrieval/assets/91877920/0fff873c-bfc7-4a62-a474-aece7ee2a22b)



# Hypergraph Propagation and Community Selection for Objects Retrieval
This is the code of our NeurIPS2021 paper: Hypergraph Propagation and Community Selection for Objects Retrieval. You can check the detail from our project page: https://sgvr.kaist.ac.kr/~guoyuan/hypergraph_propagation/ . 

```bibtex
@article{an2021hypergraph,
  title={Hypergraph propagation and community selection for objects retrieval},
  author={An, Guoyuan and Huo, Yuchi and Yoon, Sung-Eui},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={3596--3608},
  year={2021}
}

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
We only have the CPU version currently. We used several CPUs and a large amount of memory to get the result on R1M distractors. However, the complexity of the offline process is O(1). So GPU will accelerate this process a lot theoretically.
We would be grateful if anyone offered a (or introduced an existing) GPU version geometric-verification code.
