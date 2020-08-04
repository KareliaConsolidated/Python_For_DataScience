# Import Necessary Modules
import pandas as pd

df = pd.read_table('Datasets/SMSSpamCollection.txt', sep='\t', names=['label', 'sms_message'])
print(df.head(),'\n')

print('#'*100)

df['label'] = df.label.map({'ham': 0, 'spam': 1})

print(df.head(),'\n')

print('#'*100)

############### LOWER CASE #################

documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_documents = []
for i in documents:
	lower_case_documents.append(i.lower())
print(lower_case_documents)

###############################################

print('#'*100)

############### PUNCTUATION REMOVE ############

import string

def strip_punctuation(s):
	return ''.join(c for c in s if c not in string.punctuation)

sans_punctutation_documents = []
for i in lower_case_documents:
	sans_punctutation_documents.append(strip_punctuation(i))
print(sans_punctutation_documents)

###############################################

print('#'*100)

############### TOKENIZATION #####################

preprocessed_documents = []
for i in sans_punctutation_documents:
	preprocessed_documents.append(i.split())
print(preprocessed_documents)

##################################################

print('#'*100)

################# COUNTER #####################

from collections import Counter
import pprint

frequency_list = []
for i in preprocessed_documents:
	frequency_list.append(Counter(i))
pprint.pprint(frequency_list)

##################################################

print('#'*100)

########### Implementing BOW in SCIKIT ###########

documents = ['Hello, how are you!',
                'Win money, win from home.',
                'Call me now.',
                'Hello, Call hello you tomorrow?']

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(stop_words='english')                

count_vector.fit(documents)
print(count_vector.get_feature_names())
print('#'*100)
doc_array = count_vector.transform(documents).toarray()
pprint.pprint(doc_array)
print('#'*100)
frequency_matrix = pd.DataFrame(data=doc_array, columns=count_vector.get_feature_names())
print(frequency_matrix)
##################################################

print('#'*100)

################## DATA SPLIT #####################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], df['label'], random_state=1)

print(f"Number of Rows in the Total Set: {df.shape[0]}")
print(f"Number of Rows in the Training Set: {X_train.shape[0]}")
print(f"Number of Rows in the Ttest Set: {X_test.shape[0]}")
##################################################

print('#'*100)

##################################################

# Instantiate the CountVectorizer Method
count_vector = CountVectorizer()

# Fit the Training Data and then Return the Matrix
training_data = count_vector.fit_transform(X_train)

# Transform Testing Data and Return the Matrix.
testing_data = count_vector.transform(X_test)

## BAYES THEOREM IMPLEMENTATION FROM SCRATCH ##

# Let us implement the Bayes Theorem from scratch using a simple example. Let's say we are trying to find the odds of an individual having diabetes, given that he or she was tested for it and got a positive result. 
# In the medical field, such probabilies play a very important role as it usually deals with life and death situations. 

# We assume the following:

# `P(D)` is the probability of a person having Diabetes. It's value is `0.01` or in other words, 1% of the general population has diabetes(Disclaimer: these values are assumptions and are not reflective of any medical study).
# `P(Pos)` is the probability of getting a positive test result.
# `P(Neg)` is the probability of getting a negative test result.
# `P(Pos|D)` is the probability of getting a positive result on a test done for detecting diabetes, given that you have diabetes. This has a value `0.9`. In other words the test is correct 90% of the time. This is also called the Sensitivity or True Positive Rate.
# `P(Neg|~D)` is the probability of getting a negative result on a test done for detecting diabetes, given that you do not have diabetes. This also has a value of `0.9` and is therefore correct, 90% of the time. This is also called the Specificity or True Negative Rate.

# Calculate Probability of Getting a Positive Test Result, P(POS)

# P(D)
p_diabetes = 0.01

# P(~D)
p_no_diabetes = 0.99

# Sensitivity or P(POS|D)
p_pos_diabetes = 0.90

# Specificity or P(NEG|~D)
p_neg_no_diabetes = 0.90

# P(POS)
p_pos = (p_diabetes * p_pos_diabetes) + (p_no_diabetes * (1-p_neg_no_diabetes))
print(f"The Probability of Getting a Positive Test Result P(POS) is: {p_pos:1.3f}")

##################################################

print('#'*100)

##################################################

# P(D|Pos)
p_diabetes_pos = (p_diabetes * p_pos_diabetes) / p_pos
print(f'Probability of an individual having diabetes, given that that individual got a positive test result is: {p_diabetes_pos:1.3f}') 

##################################################

print('#'*100)

##################################################
# P(Pos|~D)
p_pos_no_diabetes = 0.1

# P(~D|Pos)
p_no_diabetes_pos = (p_no_diabetes * p_pos_no_diabetes)/p_pos
print(f'Probability of an individual not having diabetes, given that that individual got a positive test result is: {p_no_diabetes_pos:1.3f}')

##################################################

print('#'*100)

### Naive Bayes implementation from scratch ###
# Let's say that we have two political parties' candidates, 'Jill Stein' of the Green Party and 'Gary Johnson' of the Libertarian Party and we have the probabilities of each of these candidates saying the words 'freedom', 'immigration' and 'environment' when they give a speech:

# Probability that Jill Stein says 'freedom': 0.1 ---------> `P(F|J)`
# Probability that Jill Stein says 'immigration': 0.1 -----> `P(I|J)`
# Probability that Jill Stein says 'environment': 0.8 -----> `P(E|J)`

# Probability that Gary Johnson says 'freedom': 0.7 -------> `P(F|G)`
# Probability that Gary Johnson says 'immigration': 0.2 ---> `P(I|G)`
# Probability that Gary Johnson says 'environment': 0.1 ---> `P(E|G)`

# And let us also assume that the probability of Jill Stein giving a speech, `P(J)` is `0.5` and the same for Gary Johnson, `P(G) = 0.5`. 

# Given this, what if we had to find the probabilities of Jill Stein saying the words 'freedom' and 'immigration'? This is where the Naive Bayes'theorem comes into play as we are considering two features, 'freedom' and 'immigration'.

# P(J)
p_j = 0.5

# P(F/J)
p_j_f = 0.1

# P(I/J)
p_j_i = 0.1

p_j_text = p_j * p_j_f * p_j_i

print(f'{p_j_text:1.3f}')

##################################################

print('#'*100)

##################################################

# P(G)
p_g = 0.5

# P(F/G)
p_g_f = 0.7

# P(I/G)
p_g_i = 0.2

p_g_text = p_g * p_g_f * p_g_i
print(f'{p_g_text:1.3f}')

##################################################

print('#'*100)

p_f_i = p_j_text + p_g_text
print(f'Probability of words freedom and immigration being said are: {p_f_i}')

##################################################

print('#'*100)

##################################################

p_j_fi = p_j_text / p_f_i
print(f'The probability of Jill Stein saying the words Freedom and Immigration: {p_j_fi:1.3f}')

##################################################

print('#'*100)

##################################################

p_g_fi = p_g_text / p_f_i
print(f'The probability of Gary Johnson saying the words Freedom and Immigration: {p_g_fi:1.3f}')

##################################################

print('#'*100)

##################################################

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
predictions = naive_bayes.predict(testing_data)

##################################################

# Now that we have made predictions on our test set, our next goal is to evaluate how well our model is doing. There are various mechanisms for doing so, but first let's do quick recap of them.

# Accuracy - measures how often the classifier makes the correct prediction. It’s the ratio of the number of correct predictions to the total number of predictions (the number of test data points).

# Precision - tells us what proportion of messages we classified as spam, actually were spam.
# It is a ratio of true positives(words classified as spam, and which are actually spam) to all positives(all words classified as spam, irrespective of whether that was the correct classification), in other words it is the ratio of

# `[True Positives/(True Positives + False Positives)]`

# Recall(sensitivity) - tells us what proportion of messages that actually were spam were classified by us as spam.
# It is a ratio of true positives(words classified as spam, and which are actually spam) to all the words that were actually spam, in other words it is the ratio of

# `[True Positives/(True Positives + False Negatives)]`

# For classification problems that are skewed in their classification distributions like in our case, for example if we had a 100 text messages and only 2 were spam and the rest 98 weren't, accuracy by itself is not a very good metric. We could classify 90 messages as not spam(including the 2 that were spam but we classify them as not spam, hence they would be false negatives) and 10 as spam(all 10 false positives) and still get a reasonably good accuracy score. For such cases, precision and recall come in very handy. These two metrics can be combined to get the F1 score, which is weighted average of the precision and recall scores. This score can range from 0 to 1, with 1 being the best possible F1 score.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print(f'Accuracy Score: {accuracy_score(y_test, predictions): 1.3f}')
print(f'Precision Score: {precision_score(y_test, predictions): 1.3f}')
print(f'Recall Score: {recall_score(y_test, predictions): 1.3f}')
print(f'F1 Score: {f1_score(y_test, predictions): 1.3f}')

# CONCLUSION

# One of the major advantages that Naive Bayes has over other classification algorithms is its ability to handle an extremely large number of features. In our case, each word is treated as a feature and there are thousands of different words. Also, it performs well even with the presence of irrelevant features and is relatively unaffected by them. The other major advantage it has is its relative simplicity. Naive Bayes' works well right out of the box and tuning it's parameters is rarely ever necessary, except usually in cases where the distribution of the data is known.

# It rarely ever overfits the data. Another important advantage is that its model training and prediction times are very fast for the amount of data it can handle. All in all, Naive Bayes' really is a gem of an algorithm!