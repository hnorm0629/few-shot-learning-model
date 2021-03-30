import numpy as np
import sklearn as sk
import h5py as hp
fp = hp.File('path/to/file.h5', 'r')

"""
1. create episodes: 5-shot, 5-way
2. random sample 5 classes out of 20
3. random sample 5 images from each class as support-set
4. random sample 15 disjoint images from each class as query-set
5. average over support set for each class (prototypes)
6. compare each query image (15x5) against prototypes
    - compare through cosine distance or function (sklearn)
    - for each query image, generate 5 scores
7. choose class with max score as the image label
8. once done with query images, compare actual labels with prediction
    a. if label == prediction, assign 1
    b. if label != prediction, assign 0
9. average over {0,1}s for all query images (numpy)
10. return the accuracy
"""

def episode_performance():
    print("HELLO")

if __name__ == '__main__':
    print("HELLO")