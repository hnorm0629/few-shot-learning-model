import h5py as hp
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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


def episode(file_pointer):
    # gather class names and choose 5 random
    class_names = list(file_pointer.keys())
    five_classes = random.sample(class_names, 5)

    # initialize support and query sets & labels
    proto_list = list()
    query_list = list()
    # l1, l2, l3, l4, l5 = 0, 1, 2, 3, 4

    # for each of five classes...
    for c in five_classes:
        # get images associated with class and shuffle order
        images = file_pointer.get(c)[:]
        images_j = list(range(0, len(images)))
        random.shuffle(images_j)

        # select disjoint support and query set
        support = list()
        query = list()
        for j in range(0, 5):
            support.append(images[images_j[j]])
        for j in range(5, 20):
            query.append(images[images_j[j]])

        # average over support set for prototype
        prototype = np.mean(support, axis=0)

        # update prototype list and query list
        proto_list.append(prototype)
        query_list.append(query)
    # END for each

    # adjust array dimensions of proto_list and query_list
    prototypes = np.asarray(proto_list).reshape([5, -1])
    all_queries = query_list[0] + query_list[1] + query_list[2] + query_list[3] + query_list[4]
    queries = np.asarray(all_queries)
    queries = np.reshape(queries, [75, -1])

    q_label = 0
    correctness = 0
    # compare query images against prototypes
    for q in queries:
        new_q = np.reshape(q, [1, -1])
        current_scores = list()
        for p in prototypes:
            new_p = np.reshape(p, [1, -1])
            score = cosine_similarity(new_q, new_p)
            current_scores.append(score)
        # END for each

        # calculate prediction
        prediction = current_scores.index(max(current_scores))

        # compare actual label with prediction
        if q_label == prediction:
            correctness += 1

        q_label += 1
    # END for each

    # compute and return accuracy
    accuracy = correctness / len(query_list)
    return accuracy


if __name__ == '__main__':

    # retrieve feature vectors
    fp = hp.File('/Users/Hannah/Desktop/???/work/spring2021/'
                 'exploreCSR/other/mini-imagenet-test_v2.h5', 'r')

    # populate list of accuracies
    accuracies = list()
    for i in range(800):
        print("Round: " + str(i))
        accuracies.append(episode(fp))

    # average across accuracies (should be around 80%)
    mean_acc = np.average(accuracies)
    print("Accuracy: " + str(mean_acc))
