# FALCON

## Name
 FALCON (Functional Active Learning with Constraints based on Criticisms and Neighborhood prototypes)

## Description
FALCON is a novel active constraint-based clustering method that integrates the strengths of prototype-based summarization and criticism-based refinement to dynamically select the best super-instances.

![alt text](https://github.com/VincentBlase/FALCON/blob/main/Image/illustration_FALCON.png?raw=true)


```
import numpy as np
from FALCON.FALCON import FALCON
from mmd_critic.kernels import RBFKernel
from FALCON.open_data import open_dataset
from FALCON.compute_score import compute_score

name='IRIS' #Enter dataset name
X,y = open_dataset(name)
print('DATASET : ', name)

max_budget = 200
sigma1 = 0.1
sigma2 = 0.1

ARI_FALCON = []
NMI_FALCON = []
silhouette_FALCON = []

clusterer_FALCON = FALCON(X,y,RBFKernel(sigma1), RBFKernel(sigma2))
clustering_FALCON, intermediate_clustering = clusterer_FALCON.cluster(n=max_budget,k_proto=2, train_indices = None)
    
ARI_FALCON, NMI_FALCON, silhouette_FALCON = compute_score(intermediate_clustering,max_budget)


<-- Print table 2 values -->

print(ARI_score[25])
print(ARI_score[50])
print(ARI_score[100])
print(ARI_score[200])

```

## Installation
We recommand to download files on python 3.10.4 (Should work with other versions)

## Usage
Active clustering with pairwise constraint

## License
For open source projects, say how it is licensed.

## Project status
Keep going
