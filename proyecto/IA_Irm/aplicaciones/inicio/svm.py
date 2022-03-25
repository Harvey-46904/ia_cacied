#implementacion de super vector machine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from matplotlib import pyplot as plt


url = './media'
df = pd.read_csv(url) 
df['diagnosis'] = df['diagnosis'].map({
    'G': 0,
    'M': 1,
    'N': 2,
    'P': 3
}) 
labels = df['diagnosis'].tolist()
df['Class'] = labels 
df = df.drop(['id', 'Unnamed: 32', 'diagnosis'],
             axis=1)  
df.head()  


target_names = ['', 'M', 'B']
df['attack_type'] = df.Class.apply(lambda x: target_names[x])
df.head()


df1 = df[df.Class == 1]
df2 = df[df.Class == 2]


plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
plt.scatter(df1['radius_mean'], df1['texture_mean'], color='green', marker='+')
plt.scatter(df2['radius_mean'], df2['texture_mean'], color='blue', marker='.')

X = df.drop(['Class', 'attack_type'], axis='columns')
X.head()

y = df.Class

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


print(len(X_train))
print(len(X_test))

model = SVC(kernel='linear')


predictions = model.predict(X_test)
print(predictions)

percentage = model.score(X_test, y_test)

from sklearn.metrics import confusion_matrix
res = confusion_matrix(y_test, predictions)
print("Confusion Matrix")
print(res)
print(f"Test Set: {len(X_test)}")
print(f"Accuracy = {percentage*100} %")