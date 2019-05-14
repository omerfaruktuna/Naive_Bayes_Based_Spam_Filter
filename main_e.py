# build_dictionary, build_features and build_labels functions are inspired from below github repo but modified
# alameenkhader/spam_classifier

import os
import numpy as np
import re
import math

learning_legitimate_class_word_size = 0
learning_legitimate_class_corpus_size = 0
learning_spam_class_word_size = 0
learning_spam_class_corpus_size = 0


# Creates dictionary from all the emails in the directory
def build_dictionary(dir, val):
    global learning_legitimate_class_word_size
    global learning_legitimate_class_corpus_size
    global learning_spam_class_word_size
    global learning_spam_class_corpus_size

    # Read the file names
    emails = os.listdir(dir)
    emails.sort()
    # Array to hold all the words in the emails
    dictionary = []

    # Collecting all words from those emails
    for email in emails:
        m = open(os.path.join(dir, email),encoding='latin-1')
        for i, line in enumerate(m):
            if i == 2:  # Body of email is only 3rd line of text file
                words = line.split()
                dictionary += words

    # Removes punctuations and non alphabets

    for i in range(8):
        # print(len(dictionary))
        for index, word in enumerate(dictionary):
            if (not word.isalpha()) or (len(word) == 1):
                del dictionary[index]

    # print(len(dictionary))

    if val == 0:
        learning_legitimate_class_word_size = len(dictionary)
    elif val == 1:
        learning_spam_class_word_size = len(dictionary)

    # We now have the array of words, whoch may have duplicate entries
    dictionary = list(set(dictionary))  # Removes duplicates

    if val == 0:
        learning_legitimate_class_corpus_size = len(dictionary)
    elif val == 1:
        learning_spam_class_corpus_size = len(dictionary)

    return dictionary


def build_features(dir, dictionary):
    # Read the file names
    emails = os.listdir(dir)
    emails.sort()
    # ndarray to have the features
    features_matrix = np.zeros((len(emails), len(dictionary)))

    # collecting the number of occurances of each of the words in the emails
    for email_index, email in enumerate(emails):
        m = open(os.path.join(dir, email),encoding='latin-1')
        for line_index, line in enumerate(m):
            if line_index == 2:
                words = line.split()
                for word_index, word in enumerate(dictionary):
                    if word in words:
                        features_matrix[email_index, word_index] = 1
                    else:
                        features_matrix[email_index, word_index] = 0
                    #features_matrix[email_index, word_index] = words.count(word)
    return features_matrix

def build_features_freq(dir, dictionary):
    # Read the file names
    emails = os.listdir(dir)
    emails.sort()
    # ndarray to have the features
    features_matrix = np.zeros((len(emails), len(dictionary)))

    # collecting the number of occurances of each of the words in the emails
    for email_index, email in enumerate(emails):
        m = open(os.path.join(dir, email),encoding='latin-1')
        for line_index, line in enumerate(m):
            if line_index == 2:
                words = line.split()
                for word_index, word in enumerate(dictionary):
                    features_matrix[email_index, word_index] = words.count(word)
    return features_matrix

def build_labels(dir):
    # Read the file names
    emails = os.listdir(dir)
    emails.sort()
    # ndarray of labels
    labels_matrix = np.zeros(len(emails))

    for index, email in enumerate(emails):
        labels_matrix[index] = 1 if re.search('spms*', email) else 0

    return labels_matrix


train_dir_legitimate = './dataset/training/legitimate'
dictionary_legitimate = build_dictionary(train_dir_legitimate, 0)
features_train_legitimate = build_features(train_dir_legitimate, dictionary_legitimate)
features_train_legitimate_freq = build_features_freq(train_dir_legitimate, dictionary_legitimate)

train_dir_spam = './dataset/training/spam'
dictionary_spam = build_dictionary(train_dir_spam, 1)
features_train_spam = build_features(train_dir_spam, dictionary_spam)
features_train_spam_freq = build_features_freq(train_dir_spam, dictionary_spam)

t1, t2 = features_train_legitimate.shape

a_legitimate = []
for i in range(len(dictionary_legitimate)):
    a_legitimate.append(0)

for i in range(t1):
    a_legitimate += features_train_legitimate[i]

a_legitimate_freq = []
for i in range(len(dictionary_legitimate)):
    a_legitimate_freq.append(0)

for i in range(t1):
    a_legitimate_freq += features_train_legitimate_freq[i]

result_args = np.argpartition(a_legitimate, -200)[-200:]

legitimate_features_weights = a_legitimate[result_args]

legitimate_features = []
for i in range(len(result_args)):
    # print(dictionary_legitimate[result_args[i]])
    legitimate_features.append(dictionary_legitimate[result_args[i]])

legitimate_dict = dict(zip(legitimate_features, legitimate_features_weights))

global_learning_dictionary = list(set(dictionary_legitimate + dictionary_spam))

# print(len(global_learning_dictionary))

# print(legitimate_dict)

labels_train_legitimate = build_labels(train_dir_legitimate)

# print(features_train_legitimate)
print("\nLegitimate features are learnt and written to file...\n")
with open('legitimate_features.txt', 'w+') as the_file:
    for i in range(len(legitimate_features)):
        the_file.write('{} - {}\n'.format(legitimate_features[i], legitimate_features_weights[i]))

print("Dataset (learning) legitimate class total word size is: {}".format(learning_legitimate_class_word_size))

print("Dataset (learning) legitimate class corpus size is: {}".format(learning_legitimate_class_corpus_size))

print("-------------------------------------------------")

t1, t2 = features_train_spam.shape

a_spam = []
for i in range(len(dictionary_spam)):
    a_spam.append(0)

for i in range(t1):
    a_spam += features_train_spam[i]


a_spam_freq = []
for i in range(len(dictionary_spam)):
    a_spam_freq.append(0)

for i in range(t1):
    a_spam_freq += features_train_spam_freq[i]

result_args = np.argpartition(a_spam, -200)[-200:]

spam_features_weights = a_spam[result_args]

spam_features = []
for i in range(len(result_args)):
    spam_features.append(dictionary_spam[result_args[i]])

spam_dict = dict(zip(spam_features, spam_features_weights))
# print(spam_dict)


labels_train_spam = build_labels(train_dir_spam)

print("\nSpam features are learnt and written to file...\n")

with open('spam_features.txt', 'w+') as the_file:
    for i in range(len(spam_features)):
        the_file.write('{} - {}\n'.format(spam_features[i], spam_features_weights[i]))

print("\nDataset (learning) spam class total word size is: {}".format(learning_spam_class_word_size))

print("Dataset (learning) spam class corpus size is: {}".format(learning_spam_class_corpus_size))

print("\n-------------------------------------------------")

print("\nLearning dataset all classes corpus size is: {}\n".format(len(global_learning_dictionary)))

all_features = list(set(legitimate_features + spam_features))
common_features = list(set(legitimate_features) & set(spam_features))

###################

with open('common_features.txt', 'w+') as the_file:
    for i in range(len(common_features)):
        the_file.write('{} - {}\n'.format((i+1), common_features[i]))

###################

P_given_legitimate_all_features = []
for i in range(len(all_features)):


    if all_features[i] in dictionary_legitimate:
        P_given_legitimate_all_features.append(
            (1 + a_legitimate_freq[dictionary_legitimate.index(all_features[i])]) / (learning_legitimate_class_word_size + len(global_learning_dictionary)))
    else:
        P_given_legitimate_all_features.append(1/(learning_legitimate_class_word_size + len(global_learning_dictionary)))


"""
    if all_features[i] in legitimate_features:
        P_given_legitimate_all_features.append((1 + legitimate_dict[all_features[i]]) / (
                    learning_legitimate_class_word_size + len(global_learning_dictionary)))
    else:
        P_given_legitimate_all_features.append(
            1 / (learning_legitimate_class_word_size + len(global_learning_dictionary)))
"""

P_given_spam_all_features = []
for i in range(len(all_features)):


    if all_features[i] in dictionary_spam:
        P_given_spam_all_features.append(
            (1 + a_spam_freq[dictionary_spam.index(all_features[i])]) / (learning_spam_class_word_size + len(global_learning_dictionary)))
    else:
        P_given_spam_all_features.append(1/(learning_spam_class_word_size + len(global_learning_dictionary)))

"""
    if all_features[i] in spam_features:
        P_given_spam_all_features.append(
            (1 + spam_dict[all_features[i]]) / (learning_spam_class_word_size + len(global_learning_dictionary)))
    else:
        P_given_spam_all_features.append(1 / (learning_spam_class_word_size + len(global_learning_dictionary)))

"""

P_given_legitimate_all_features_dict = dict(zip(all_features, P_given_legitimate_all_features))

P_given_spam_all_features_dict = dict(zip(all_features, P_given_spam_all_features))

test_dir_legitimate = './dataset/test/legitimate'
emails_legitimate = os.listdir(test_dir_legitimate)

test_dir_spam = './dataset/test/spam'
emails_spam = os.listdir(test_dir_spam)

emails_legitimate.sort()
emails_spam.sort()

print("------------------")

legitimate_candidate = math.log10(len(emails_legitimate) / (len(emails_legitimate) + len(emails_spam)))
spam_candidate = math.log10(len(emails_spam) / (len(emails_legitimate) + len(emails_spam)))

labels_test_legitimate = []
for email in emails_legitimate:
    m = open(os.path.join(test_dir_legitimate, email),encoding='latin-1')
    for i, line in enumerate(m):
        if i == 2:
            words = line.split()
            for i in range(len(words)):
                if words[i] in all_features:
                    legitimate_candidate += math.log10(P_given_legitimate_all_features_dict[words[i]])
                    spam_candidate += math.log10(P_given_spam_all_features_dict[words[i]])
    if legitimate_candidate > spam_candidate:
        labels_test_legitimate.append(0)
    else:
        labels_test_legitimate.append(1)

    legitimate_candidate = math.log10(len(emails_legitimate) / (len(emails_legitimate) + len(emails_spam)))
    spam_candidate = math.log10(len(emails_spam) / (len(emails_legitimate) + len(emails_spam)))

print("Performance in legitimate test dataset: \n")

print("Number of test data: {}".format(len(labels_train_legitimate)))

print("Number of correctly predicted data: {}".format(labels_test_legitimate.count(0)))

print("Precision in legitimate test dataset is %{:.2f}".format(
    labels_test_legitimate.count(0) / len(labels_train_legitimate) * 100))

print("------------------")

legitimate_candidate = math.log10(len(emails_legitimate) / (len(emails_legitimate) + len(emails_spam)))
spam_candidate = math.log10(len(emails_spam) / (len(emails_legitimate) + len(emails_spam)))

labels_test_spam = []
for email in emails_spam:
    m = open(os.path.join(test_dir_spam, email),encoding='latin-1')
    for i, line in enumerate(m):
        if i == 2:
            words = line.split()
            for i in range(len(words)):
                if words[i] in all_features:
                    legitimate_candidate += math.log10(P_given_legitimate_all_features_dict[words[i]])
                    spam_candidate += math.log10(P_given_spam_all_features_dict[words[i]])
    if legitimate_candidate > spam_candidate:
        labels_test_spam.append(0)
    else:
        labels_test_spam.append(1)

    legitimate_candidate = math.log10(len(emails_legitimate) / (len(emails_legitimate) + len(emails_spam)))
    spam_candidate = math.log10(len(emails_spam) / (len(emails_legitimate) + len(emails_spam)))

print("Performance in spam test dataset: \n")

print("Number of test data: {}".format(len(labels_train_spam)))
print("Number of correctly predicted data: {}".format(labels_test_spam.count(1)))

print("Precision in spam test dataset is %{:.2f}".format(labels_test_spam.count(1) / len(labels_train_spam) * 100))

TP1 = labels_test_legitimate.count(0)
FP1 = len(labels_train_legitimate) - labels_test_legitimate.count(0)
FN1 = len(labels_train_spam) - labels_test_spam.count(1)
TN1 = labels_test_spam.count(1)

TP2 = labels_test_spam.count(1)
FP2 = len(labels_train_spam) - labels_test_spam.count(1)
FN2 = len(labels_train_legitimate) - labels_test_legitimate.count(0)
TN2 = labels_test_legitimate.count(0)


Precision_1 = TP1/(TP1+FP1)
Recall_1 = TP1/(TP1+FN1)
F_measure_1 = 2*Precision_1*Recall_1/(Precision_1+Recall_1)

Precision_2 = TP2/(TP2+FP2)
Recall_2 = TP2/(TP2+FN2)
F_measure_2 = 2*Precision_2*Recall_2/(Precision_2+Recall_2)

macro_averaged_precision = (Precision_1 + Precision_2)/2
micro_averaged_precision = (TP1 + TP2) / (TP1+FP1+TP2+FP2)

macro_averaged_recall = (Recall_1+Recall_2) / 2
micro_averaged_recall = (TP1 + TP2) / (TP1+FN1+TP2+FN2)

print("\n------------------\n")

print("Precision for legitimate mail test set is: %{:.4f} percent.".format(Precision_1*100))
print("Recall for legitimate mail test set is: %{:.4f} percent.".format(Recall_1*100))
print("F_Measure for legitimate mail test set is: %{:.4f} percent.".format(F_measure_1*100))

print("Precision for spam mail test set is: %{:.4f} percent.".format(Precision_2*100))
print("Recall for spam mail test set is: %{:.4f} percent.".format(Recall_2*100))
print("F_Measure for spam mail test set is: %{:.4f} percent.".format(F_measure_2*100))

print("Macro Averaged Precision is: %{:.4f} percent.".format(macro_averaged_precision*100))
print("Micro Averaged Precision is: %{:.4f} percent.".format(micro_averaged_precision*100))

print("Macro Averaged Recall is: %{:.4f} percent.".format(macro_averaged_recall*100))
print("Micro Averaged Recall is: %{:.4f} percent.".format(micro_averaged_recall*100))
