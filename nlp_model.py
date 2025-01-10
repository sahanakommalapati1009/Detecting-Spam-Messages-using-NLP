import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import pickle

#Loading the dataset spam.csv contains labeled text messages
df= pd.read_csv("spam.csv", encoding="latin-1")

#The dataset has unused columns (Unnamed: 2, Unnamed: 3, Unnamed: 4), which are dropped to clean up the data.
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Features and Labels
# The class column contains labels (ham for legitimate messages, spam for junk). These are mapped to numeric values.
df['label'] = df['class'].map({'ham': 0, 'spam': 1})

# X is the independent variable, holds the text data (message column).
X = df['message']

# Y is the dependent variable,  holds the numeric labels (label column).
y = df['label']

# Extract Feature With CountVectorizer
# procedure : Example: ["I love programming.", "I love Python."]
# Splits the text into individual words (tokens).
# Identifies all unique words in the dataset 
# ['I', 'love', 'programming', 'Python']
# For each document, it creates a vector where:
# -Each dimension represents a word from the vocabulary.
# -The value in each dimension is the count of the word in the document.
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data

# Save the CountVectorizer object using pickle for reuse
pickle.dump(cv, open('tranform.pkl', 'wb'))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

# A MultinomialNB model (Naive Bayes classifier) is used. It is well-suited for text classification tasks.
clf = MultinomialNB()
#The model is trained on the X_train and y_train data.
clf.fit(X_train,y_train)
# The model's accuracy is calculated using X_test and y_test.
clf.score(X_test,y_test)

# The trained Naive Bayes model is saved as a .pkl file using pickle for reuse in other programs.
filename = 'nlp_model.pkl'
pickle.dump(clf, open(filename, 'wb'))

#Saves the trained model in an efficient format.
joblib.dump(clf, 'NB_spam_model.pkl')
NB_spam_model = open('NB_spam_model.pkl','rb')

# Loads the saved model for predictions or further evaluation
clf = joblib.load(NB_spam_model)