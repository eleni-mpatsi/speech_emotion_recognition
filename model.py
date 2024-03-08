import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


#φορτώνουμε το αρχείο pickle ως dataframe 
df = pd.read_pickle('prepared_dataframe.pickle')

#αφαιρούμε τις μηδενικές τιμές για την καλυτερη επίδοση του μοντέλου
df = df.replace(0, np.nan)  # Replace zeros with NaN
df = df.dropna()  # Drop rows with NaN values

'''αντιστοιχίζουμε τους αριθμούς στα ανάλογα συναισθήματα , τα οποία θα 
αποτελέσουν τα target labels μας '''
emotion_labels = {
    1: 'neutral',
    2: 'calm',
    3: 'happy',
    4: 'sad',
    5: 'angry',
    6: 'fearful',
    7: 'disgust',
    8: 'surprised'
}

#μετατρέπουμε τις τιμές της στήλης σε integers με την astype()
df['emotion'] = df['emotion'].astype(int)

'''αντικαθιστούμε τους αριθμούς της στήλης με τα emotion labels 
όπως τα ορίσαμε παραπάνω '''
df['emotion'] = df['emotion'].replace(emotion_labels)

'''δημιουργούμε ένα νέο dataframe (filtered_df) ,για να απομονώσουμε 
τα δύο συγκεκριμένα συναισθήματα που μας ενδιαφέρουν για το classification '''
filtered_df = df[df['emotion'].isin(['calm', 'angry'])]


'''καθώς το dataframe ήταν ελαφρώς imbalanced , αποφασίσαμε να κάνουμε 
downsampling , για να αποφύγουμε balanced προβλεψεις'''
label_counts = filtered_df['emotion'].value_counts()
min_count = label_counts.min()
downsampled_df = pd.concat([filtered_df[filtered_df['emotion'] == label].sample(n=min_count, replace=False, random_state=42) for label in label_counts.index])

'''ορίζουμε τα features για το training (X) και τα target labels για τις
προβλέψεις (y) '''
X = downsampled_df[['mean_centroid', 'std_centroid', 'mean_mfccs', 'std_mfccs', 'mean_bandwidth', 'std_bandwidth']]
y = downsampled_df['emotion']

#χωρίζουμε τα δεδομένα μας σε train/test με την train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#εκπαιδεύουμε ένα Logistic Regression μοντέλο 
model = LogisticRegression()
model.fit(X_train, y_train)

print('Logistic Regression Model')

#κάνουμε προβλέψεις για το test set 
y_pred = model.predict(X_test)
print("Predictions:", y_pred)

'''υπολογίζουμε τις βασικές μετρικές , για να ελέγξουμε την επίδοση του 
μοντέλου (accuracy , precision , recall , f1 score , confusion matrix)'''
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
precision = precision_score(y_test, y_pred, average='weighted')
print("Precision:", precision)
recall = recall_score(y_test, y_pred, average='weighted')
print("Recall:", recall)
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1-score:", f1)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# αποθηκεύουμε το μοντέλο με την joblib.dump 
joblib.dump(model, 'emotion_classification_model.pkl')

