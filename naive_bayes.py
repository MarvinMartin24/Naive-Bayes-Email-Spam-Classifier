
import os
import numpy as np
import string
from collections import defaultdict
import pandas as pd

# Return a trainset and testset dataframe from filename
def create_datasets(doc_dir, split):
    df = pd.read_csv(doc_dir, sep="\t", names=["Y","X"], header=None)
    mask = np.random.rand(len(df)) < split
    train = df[mask]
    test = df[~mask]
    return train, test

# Return a list of tuple of size N that represent top words frequencies
def make_Dictionary(trainset, size):
    dictionary = defaultdict(int)
    for mail in trainset["X"]:
        # Delete headspace and linebreak character, move to lowercase and remove the punctuation
        mail = mail.strip().lower().translate(mail.maketrans("", "", string.punctuation))
        words = mail.split(" ")
        for word in words:
            # All characters in the string are alphabet
            if word.isalpha():
                dictionary[word] += 1

    final_dictionary = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)[:size]
    return final_dictionary

# Return a numpy array of the vocabulary dictionary size that encoded a string
def sentence2vec(vocabulary, mail):
    # vocabulary as a list, not tuple
    vocabulary = [item[0] for item in vocabulary]
    # Initialize feature vector
    features_vector = np.zeros((len(vocabulary)))
    # Delete headspace and linebreak character, move to lowercase and remove the punctuation
    mail = mail.strip().lower().translate(mail.maketrans("", "", string.punctuation))
    words = mail.split(" ")
    for word in words:
        if word.isalpha():
            if word in vocabulary:
                features_vector[vocabulary.index(word)] += 1
    return features_vector

# Binary Encoding for Label
def to_categorical(str_y):
    if str_y == "spam":
        return 1
    else:
        return 0

# return P(Y), P(X|Y=0), P(X|Y=0) based on the Trainset dataframe
def train(dfX, dfY, alpha):
    # From list dataframe to numpy matrix
    X = np.array(dfX.tolist())
    Y = np.array(dfY)

    numTrainMails = X.shape[0]
    numWords = X.shape[1]

    p_spam = sum(Y)/float(numTrainMails) #P(Y=1)
    # Create a vector of the size of the vocabulary, to compute the probably of the appearance of a word given it is a spam or ham
    # To prevent 0/0 we Initialize our probabilistic vector to a small constante alpha
    p0Num, p1Num = np.full((numWords), alpha), np.full((numWords), alpha)
    p0Den, p1Den = alpha , alpha

    # For each training mail
    for i in range(numTrainMails):
        # if the mail is a spam
        if Y[i] == 1:
            # Increase the counter of the seen word (here the sum is between to vector -> counter = [0,..,0] | mailvect = [1,..,0] => counter+= mailvect)
            p1Num += X[i]
            # Total number of word of the given mail is summed up and added to p1Den
            p1Den += sum(X[i])
        # if the mail is a ham
        else:
            p0Num += X[i]
            # Total number of word of the given mail is summed up and added to p0Den
            p0Den += sum(X[i])

    # Compute a vector of size numWords, which include the conditional log probabilty of every single world
    p1Vector = np.log(p1Num/p1Den) #P(X|Y=1)
    p0Vector = np.log(p0Num/p0Den) #P(X|Y=0)

    return p_spam, p0Vector, p1Vector

# Return the predicted label based on the train function output and the new mail feature vector input
def classifier(mail2vec, p_spam, p0Vector, p1Vector):
    p1 = sum(mail2vec * p1Vector) + np.log(p_spam) #P(Y=1|X)
    p0 = sum(mail2vec * p0Vector) + np.log(1 - p_spam) #P(Y=0|X)
    if p1 > p0:
        return 1 # It a spam
    else:
        return 0 # It a ham

# Return confusion matrix given prediction and label (in two dataframes)
def get_confusion_matrix(dfY, dfY_pred):
    # Convert dataframes to arrays
    Y = np.array(dfY.tolist())
    Y_pred = np.array(dfY_pred.tolist())

    numClasses = len(np.unique(Y)) # Number of classes
    matrix = np.zeros((numClasses, numClasses))
    TP, FP, FN, TN = 0, 0, 0, 0

    for i in range(len(Y)):
        if Y_pred[i] == 1 and Y[i] == 1:
            TP += 1
        if Y_pred[i] == 0 and Y[i] == 0:
            TN += 1
        if Y_pred[i] == 1 and Y[i] == 0:
            FP += 1
        if Y_pred[i] == 0 and Y[i] == 1:
            FN += 1

        matrix[1 - Y_pred[i]][1 - Y[i]] += 1
    return TP, FP, FN, TN, matrix


if __name__ == '__main__':


    trainset, testset = create_datasets("messages.txt", split=0.8)


    print("\n\nGenerate a dictionary from the training data.")
    vocabulary = make_Dictionary(trainset, size=3000)
    print("     Vocabulary Dictionary (sample of the top 15 most seen words):s")
    print(vocabulary[:15])

    print("\n\nExtract features from both the training data and test data.")
    print("\n   Final Trainset with encoded data (sample of 15 examples)")
    trainset["X"] = trainset["X"].apply(lambda x: sentence2vec(vocabulary, x))
    trainset["Y"] = trainset["Y"].apply(lambda y: to_categorical(y))
    print(trainset.shape)
    print(trainset.sample(15))

    print("\n   Final Testset with encoded data (sample of 15 examples)")
    testset["X"] = testset["X"].apply(lambda x: sentence2vec(vocabulary, x))
    testset["Y"] = testset["Y"].apply(lambda y: to_categorical(y))
    print(testset.shape)
    print(testset.sample(15))


    print(f"    We picked alpha = {0.001}")
    p_spam, p0Vector, p1Vector = train(trainset["X"], trainset["Y"], alpha = 0.001)
    print(f"    Probability to get a spam over all train mails: {p_spam}")
    print(f"    Vector of conditional log Probability for every word of the vocabulary given a spam:\n {p1Vector}")
    print(f"    Vector of conditional log Probability for every word of the vocabulary given a ham:\n {p0Vector}")

    # Single test on a fake mail
    # fake_mail =  "Hello, World"
    # mail2vec = sentence2vec(vocabulary, fake_mail)
    # pred = classifier(mail2vec, p_spam, p0Vector, p1Vector)
    # print(pred)

    testset["Y_pred"] = testset["X"].apply(lambda y: classifier(y, p_spam, p0Vector, p1Vector))
    numSpam = sum(testset["Y_pred"])
    totalTestMail = len(testset["Y_pred"])

    print("\n\n Measure the spam-filtering performance for each approach through the confusion matrix.")
    TP, FP, FN, TN, confusion_matrix = get_confusion_matrix(testset["Y"],testset["Y_pred"])
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    print(f"    Y_pred (rows) vs Y (columns)\n  |    1    |    0    |\n1 | TP = {TP}| FP = {FP} |\n0 | FN = {FN} | TN = {TN}|")
    print(f"    In the testset there is {numSpam} spam out of {totalTestMail} mails")
    print(f"    precision = {precision}% | recall = {recall}% | accuracy = {accuracy}%")
