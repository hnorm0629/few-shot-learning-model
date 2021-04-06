import h5py as hp
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


def episode(file_pointer):
    # gather class names and choose 5 random
    class_names = list(file_pointer.keys())
    five_classes = random.sample(class_names, 5)

    # initialize support and query sets & labels
    proto_list = list()
    query_list = list()

    # for each of five classes...
    for c in five_classes:
        # get images associated with class and shuffle order
        images = file_pointer.get(c)[:]

        # average over last two dimensions of images
        im_mean = np.mean(images, axis=(2, 3))         # change dimensionality

        images_j = list(range(0, len(images)))
        random.shuffle(images_j)

        # select disjoint support and query set
        support = list()
        query = list()
        for j in range(0, 5):                          # change to 1 for 1-shot
            support.append(im_mean[images_j[j]])
        for j in range(5, 20):
            query.append(im_mean[images_j[j]])

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

    count = 0
    label = 0
    correctness = 0
    # compare query images against prototypes
    for q in queries:
        new_q = np.reshape(q, [1, -1])
        current_scores = list()
        for p in prototypes:
            new_p = np.reshape(p, [1, -1])
            score = cosine_similarity(new_q, new_p)             # change to euclidean_distances
            current_scores.append(score)
        # END for each

        # calculate prediction
        prediction = current_scores.index(max(current_scores))  # if euclidean, change to min

        # compare actual label with prediction
        if label == prediction:
            correctness += 1

        # keep track of actual label
        count += 1
        if count % 15 == 0:
            label += 1
    # END for each

    # compute and return accuracy
    accuracy = correctness / len(queries)
    return accuracy


if __name__ == '__main__':
    # retrieve feature vectors
    fp = hp.File('/Users/Hannah/Desktop/???/work/spring2021/'
                 'exploreCSR/other/mini-imagenet-test_v2.h5', 'r')

    # populate list of accuracies
    accuracies = list()
    for i in range(1, 801):
        if i % 5 == 0:
            print("Episode: " + str(i))  # display counter
        accuracies.append(episode(fp))

    # average across accuracies
    mean_acc = np.average(accuracies)
    print("Accuracy: " + str(round(mean_acc, 6)) +
          " = " + f"{mean_acc:.2%}")
