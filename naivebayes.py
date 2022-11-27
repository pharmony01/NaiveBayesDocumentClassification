# naivebayes.py
"""Perform document classification using a Naive Bayes model."""

import argparse
import os
import pdb
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

ROOT = (r'C:\Users\riley\Desktop\CSC 3520 HW1\data')  # change to path where data is stored

parser = argparse.ArgumentParser(description="Use a Naive Bayes model to classify text documents.")
parser.add_argument('-x', '--training_data',
                    help='path to training data file, defaults to ROOT/trainingdata.txt',
                    default=os.path.join(ROOT, 'trainingdata.txt'))
parser.add_argument('-y', '--training_labels',
                    help='path to training labels file, defaults to ROOT/traininglabels.txt',
                    default=os.path.join(ROOT, 'traininglabels.txt'))
parser.add_argument('-xt', '--testing_data',
                    help='path to testing data file, defaults to ROOT/testingdata.txt',
                    default=os.path.join(ROOT, 'testingdata.txt'))
parser.add_argument('-yt', '--testing_labels',
                    help='path to testing labels file, defaults to ROOT/testinglabels.txt',
                    default=os.path.join(ROOT, 'testinglabels.txt'))
parser.add_argument('-n', '--newsgroups',
                    help='path to newsgroups file, defaults to ROOT/newsgroups.txt',
                    default=os.path.join(ROOT, 'newsgroups.txt'))
parser.add_argument('-v', '--vocabulary',
                    help='path to vocabulary file, defaults to ROOT/vocabulary.txt',
                    default=os.path.join(ROOT, 'vocabulary.txt'))


def main(args):
    print("Document Classification using Naïve Bayes Classifiers")
    print("=======================")
    print("PRE-PROCESSING")
    print("=======================")

    # Parse input arguments
    training_data_path = os.path.expanduser(args.training_data)
    training_labels_path = os.path.expanduser(args.training_labels)
    testing_data_path = os.path.expanduser(args.testing_data)
    testing_labels_path = os.path.expanduser(args.testing_labels)
    newsgroups_path = os.path.expanduser(args.newsgroups)
    vocabulary_path = os.path.expanduser(args.vocabulary)

    # Load data from relevant files
    # ***MODIFY CODE HERE***
    print("Loading training data...")
    xtrain = np.loadtxt(training_data_path, dtype=int)
    print("Loading training labels...")
    ytrain = np.loadtxt(training_labels_path, dtype=int)
    print("Loading testing data...")
    xtest = np.loadtxt(testing_data_path, dtype=int)
    print("Loading testing labels...")
    ytest = np.loadtxt(testing_labels_path, dtype=int)
    print("Loading newsgroups...")
    newsgroups = np.loadtxt(newsgroups_path, dtype=str)
    print("Loading vocabulary...")
    vocabulary = np.loadtxt(vocabulary_path, dtype=str)

    # The order is DocID, WordID, Count
    # Change 1-indexing to 0-indexing for labels, docID, wordID
    # ***MODIFY CODE HERE***
    # Opted not to do this


    # Extract useful parameters
    num_training_documents = len(ytrain)
    num_testing_documents = len(ytest)
    num_words = len(vocabulary)
    num_newsgroups = len(newsgroups)

    print("\n=======================")
    print("TRAINING")
    print("=======================")

    # Estimate the prior probabilities
    print("Estimating prior probabilities via MLE...")
    # ***MODIFY CODE HERE***
    _,priors = np.unique(ytrain, return_counts=True)
    priors = priors / num_training_documents
    
    # Estimate the class conditional probabilities
    print("Estimating class conditional probabilities via MAP...")
    # ***MODIFY CODE HERE***

    # Creates a 2d array of width number of labels and height number of documents
    class_conditionals = np.zeros([num_words, num_newsgroups])
    for arr in xtrain:
        class_conditionals[arr[1] - 1][ytrain[arr[0] - 1] - 1] += arr[2]
    
    beta = 1/num_words

    for col in range(0, len(class_conditionals[0])):
        class_conditionals[:,col] += beta


    print("\n=======================")
    print("TESTING")
    print("=======================")

    # Test the Naive Bayes classifier
    print("Applying natural log to prevent underflow...")
    # ***MODIFY CODE HERE***
    for col in range(0, len(class_conditionals[0])):
        total_words = np.sum(class_conditionals[:,col])  
        class_conditionals[:,col] = np.log(class_conditionals[:,col])  - np.log(total_words)
        priors[col] = np.log(priors[col])


    print("Counting words in each document...")
    # ***MODIFY CODE HERE***
    counts = np.zeros([num_words, num_testing_documents])
    for arr in xtest:
        counts[arr[1] - 1][arr[0] - 1] = arr[2]

    print("Computing posterior probabilities...")
    # ***MODIFY CODE HERE***
    log_posterior = np.zeros([num_testing_documents,num_newsgroups])
    t_counts = np.transpose(counts)
    log_posterior = priors + np.matmul(t_counts, class_conditionals)

    print("Assigning predictions via argmax...")
    # ***MODIFY CODE HERE***
    pred = np.argmax(log_posterior, axis=1)
    pred += 1
    print("\n=======================")
    print("PERFORMANCE METRICS")
    print("=======================")

    # Compute performance metrics
    # ***MODIFY CODE HERE***

    accuracy = np.count_nonzero(np.equal(pred, ytest)) / num_testing_documents * 100
    print(f"Accuracy: {accuracy:.2f}%")
    cm = confusion_matrix(ytest, pred)
    print("Confusion matrix:")
    print(cm)

    # pdb.set_trace()  # uncomment for debugging, if needed


if __name__ == '__main__':
    main(parser.parse_args())
