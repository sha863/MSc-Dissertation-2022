print('\033[1mLogistic Regression Classifier using counting word occurence   \033[0m')

# Split Dataset for training and testing
x_train,x_test,y_train,y_test = train_test_split(data['text'], data.label, test_size=0.2, random_state=41)

pipe = Pipeline([('vect', CountVectorizer()), # Converts text into numeric format to feed in models
                  ('model', LogisticRegression())]) # Model Classifier

# Fitting of pipeline
model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)

print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

x_axis_labels = ["ham", "spam"]
y_axis_labels = ["ham", "spam"] 

sns.heatmap(confusion_matrix(y_test, prediction), annot=True, fmt="d", cmap='BuPu', xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("true labels")
plt.ylabel("predicted label")

plt.show()
print("\n")
print("\t\tClassification report\n")
print(classification_report(y_test, prediction))
print("\n------------------------------------------------------------------------------------------\n")

print('\033[1mLogistic Regression Classifier using normalized count occurence   \033[0m')
# Split Dataset for training and testing
x_train,x_test,y_train,y_test = train_test_split(data['text'], data.label, test_size=0.2, random_state=41)

pipe = Pipeline([('vect', TfidfVectorizer(use_idf=False, norm="l2")), # Converts text into numeric format to feed in models
                  ('model', LogisticRegression())]) # Model Classifier

# Fitting of pipeline
model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

x_axis_labels = ["ham", "spam"]
y_axis_labels = ["ham", "spam"] 

sns.heatmap(confusion_matrix(y_test, prediction), annot=True, fmt="d", cmap='BuPu', xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("true labels")
plt.ylabel("predicted label")

plt.show()
print("\n")
print("\t\tClassification report\n")
print(classification_report(y_test, prediction))
print("\n--------------------------------------------------------------------------------------------\n")

print('\033[1mLogistic Regression Classifier Classifier using TFIDF   \033[0m')
# Split Dataset for training and testing
x_train,x_test,y_train,y_test = train_test_split(data['text'], data.label, test_size=0.2, random_state=41)

pipe = Pipeline([('vect', CountVectorizer()), # Converts text into numeric format to feed in models
                 ('tfidf', TfidfTransformer()), # Technique to extract features from data
                 ('model', LogisticRegression())]) # Model Classifier

# Fitting of pipeline
model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

x_axis_labels = ["ham", "spam"]
y_axis_labels = ["ham", "spam"] 

sns.heatmap(confusion_matrix(y_test, prediction), annot=True, fmt="d", cmap='BuPu', xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("true labels")
plt.ylabel("predicted label")

plt.show()
print("\n")
print("\t\tClassification report\n")
print(classification_report(y_test, prediction))

print("\n------------------------------------------------------------------------------------------\n")
print('\033[1mLogistic Regression Classifier using counting word occurence with 2-grams   \033[0m')

# Split Dataset for training and testing
x_train,x_test,y_train,y_test = train_test_split(data['text'], data.label, test_size=0.2, random_state=41)

pipe = Pipeline([('vect', CountVectorizer(ngram_range=(2,2))), # Converts text into numeric format to feed in models
                ('model', LogisticRegression())]) # Model Classifier

# Fitting of pipeline
model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)

print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

x_axis_labels = ["ham", "spam"]
y_axis_labels = ["ham", "spam"] 

sns.heatmap(confusion_matrix(y_test, prediction), annot=True, fmt="d", cmap='BuPu', xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("true labels")
plt.ylabel("predicted label")

plt.show()
print("\n")
print("\t\tClassification report\n")
print(classification_report(y_test, prediction))
print("\n------------------------------------------------------------------------------------------\n")
