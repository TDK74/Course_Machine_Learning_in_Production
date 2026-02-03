import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from statistics import mean
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


## ------------------------------------------------------ ##
def load_labeled_spam_dataset():
    base_path = "./data"

    csv_files = [os.path.join(base_path, csv) for csv in os.listdir(base_path)]

    dfs = [pd.read_csv(filename) for filename in csv_files]

    df = pd.concat(dfs)

    df = df.rename(columns = {"CONTENT" : "text", "CLASS" : "label"})

    df = df.sample(frac = 1, random_state = 423)

    return df.reset_index()

df_labeled = load_labeled_spam_dataset()

## ------------------------------------------------------ ##
df_labeled.head()

## ------------------------------------------------------ ##
print(f"Value counts for each class:\n\n{df_labeled.label.value_counts()}\n")

df_labeled.label.value_counts().plot.pie(y = 'label', title = 'Proportion of each class')
plt.show()

## ------------------------------------------------------ ##
df_labeled = df_labeled.drop(['index', 'COMMENT_ID', 'AUTHOR', 'DATE'], axis = 1)

df_labeled.head()

## ------------------------------------------------------ ##
X = df_labeled.drop("label", axis = 1)

y = df_labeled["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42,
                                                    stratify = y)

print(f"There are {X_train.shape[0]} comments for training.")
print(f"There are {X_test.shape[0]} comments for testing")

## ------------------------------------------------------ ##
plt.subplot(1, 3, 1)
y_train.value_counts().plot.pie(y = 'label', title = 'Proportion of each class for train set',
                                figsize = (10, 6))

plt.subplot(1, 3, 3)
y_test.value_counts().plot.pie(y = 'label', title = 'Proportion of each class for test set',
                                figsize = (10, 6))

plt.tight_layout()
plt.show()

## ------------------------------------------------------ ##
vectorizer = CountVectorizer(ngram_range = (1, 5))

## ------------------------------------------------------ ##
def calculate_accuracy(X_tr, y_tr, X_te = X_test, y_te = y_test, clf = MultinomialNB(),
                        vectorizer = vectorizer):
    X_train_vect = vectorizer.fit_transform(X_tr.text.tolist())

    clf.fit(X = X_train_vect, y = y_tr)

    X_test_vect = vectorizer.transform(X_te.text.tolist())

    preds = clf.predict(X_test_vect)

    return accuracy_score(preds, y_te)

## ------------------------------------------------------ ##
accs = dict()

## ------------------------------------------------------ ##
rnd_labels = np.random.randint(0, 2, X_train.shape[0])

rnd_acc = calculate_accuracy(X_train, rnd_labels)

rnd_acc   # print()

## ------------------------------------------------------ ##
rnd_accs = []

for _ in range(10):
    rnd_accs.append(calculate_accuracy(X_train, np.random.randint(0, 2, X_train.shape[0])))

accs['random-labels'] = sum(rnd_accs) / len(rnd_accs)

print(f"The random labeling method achieved an accuracy of {accs['random-labels'] * 100:.2f}%")

## ------------------------------------------------------ ##
true_acc = calculate_accuracy(X_train, y_train)

accs['true-labels'] = true_acc

print(f"The true labeling method achieved an accuracy of {accs['true-labels'] * 100:.2f}%")

## ------------------------------------------------------ ##
def labeling_rules_1(x):
    x = x.lower()

    rules = ["free" in x, "subs" in x, "http" in x]

    if any(rules):
        return 1

    return -1

## ------------------------------------------------------ ##
labels = [labeling_rules_1(label) for label in X_train.text]

labels = np.asarray(labels)

labels   # print()

## ------------------------------------------------------ ##
X_train_al = X_train[labels != -1]

labels_al = labels[labels != -1]

print(f"Predictions with concrete label have shape: {labels_al.shape}")

print(f"Proportion of data points kept: {labels_al.shape[0] / labels.shape[0] * 100:.2f}%")

## ------------------------------------------------------ ##
iter_1_acc = calculate_accuracy(X_train_al, labels_al)

print(f"First iteration of automatic labeling has an accuracy of {iter_1_acc * 100:.2f}%")

accs['first-iteration'] = iter_1_acc

## ------------------------------------------------------ ##
def plot_accuracies(accs = accs):
    colors = list("rgbcmy")
    items_num = len(accs)
    cont = 1

    for x, y in accs.items():
        if x in ['true-labels', 'random-labels', 'true-labels-best-clf']:
            plt.hlines(y, 0, (items_num - 2) * 2, colors = colors.pop())

        else:
            plt.scatter(cont, y, s = 100)
            cont += 2

    plt.legend(accs.keys(), loc = "center left", bbox_to_anchor = (1, 0.5))
    plt.show()

plot_accuracies()

## ------------------------------------------------------ ##
def label_given_rules(df, rules_function, name, accs_dict = accs, verbose = True):
    labels = [rules_function(label) for label in df.text]

    labels = np.asarray(labels)

    initial_size = labels.shape[0]

    X_train_al = df[labels != -1]
    labels = labels[labels != -1]

    final_size = labels.shape[0]

    acc = calculate_accuracy(X_train_al, labels)

    if verbose:
        print(f"Proportion of data points kept: {final_size / initial_size * 100:.2f}%\n")
        print(f"{name} labeling has an accuracy of {acc * 100:.2f}%\n")

    accs_dict[name] = acc

    return X_train_al, labels, acc

## ------------------------------------------------------ ##
def labeling_rules_2(x):
    x = x.lower()

    not_spam_rules = ["view" in x, "song" in x]

    spam_rules = ["free" in x, "subs" in x, "gift" in x, "follow" in x, "http" in x]

    if any(not_spam_rules):
        return 0

    if any(spam_rules):
        return 1

    return -1

## ------------------------------------------------------ ##
label_given_rules(X_train, labeling_rules_2, "second-iteration")

plot_accuracies()

## ------------------------------------------------------ ##
print(f"NOT_SPAM comments have an average of {mean(
                        [len(t) for t in df_labeled[df_labeled.label == 0].text]):.2f} characters.")
print(f"SPAM comments have an average of {mean(
                        [len(t) for t in df_labeled[df_labeled.label == 1].text]):.2f} characters.")

# or if modify the original rows:
# mean_not_spam = mean([len(t) for t in df_labeled[df_labeled.label == 0].text])
# print(f"NOT_SPAM comments have an average of {mean_not_spam:.2f} characters.")

# mean_spam = mean([len(t) for t in df_labeled[df_labeled.label == 1].text])
# print(f"SPAM comments have an average of {mean_spam:.2f} characters.")

## ------------------------------------------------------ ##
plt.hist([len(t) for t in df_labeled[df_labeled.label == 0].text], range = (0, 100))
plt.show()

## ------------------------------------------------------ ##
def labeling_rules_3(x):
    x = x.lower()

    not_spam_rules = ["view" in x, "song" in x, len(x) < 30]

    spam_rules = ["free" in x, "subs" in x, "gift" in x,
                "follow" in x, "http" in x, "check out" in x]

    if any(not_spam_rules):
        return 0

    if any(spam_rules):
        return 1

    return -1

## ------------------------------------------------------ ##
label_given_rules(X_train, labeling_rules_3, "third-iteration")

plot_accuracies()

## ------------------------------------------------------ ##
pd.set_option('display.max_rows', None)

df_labeled[df_labeled.label == 0]

## ------------------------------------------------------ ##
def your_labeling_rules(x):
    x = x.lower()

    # Define your rules for classifying as NOT_SPAM
    not_spam_rules = [
#                   "view" in x, "song" in x, len(x) < 30
                    ]

    # Define your rules for classifying as SPAM
    spam_rules = [
#               "free" in x, "subs" in x, "gift" in x, "follow" in x, "http" in x, "check out" in x
                ]

    if any(not_spam_rules):
        return 0

    if any(spam_rules):
        return 1

    return -1


try:
    label_given_rules(X_train, your_labeling_rules, "your-iteration")
    plot_accuracies()

except ValueError:
    print("You have not defined any rules.")
