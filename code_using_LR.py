import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
df = pd.read_csv('Finance_data.csv')
df = df.drop(columns=['Unnamed: 0'], errors='ignore')
df.fillna(df.select_dtypes(include = 'number').median(), inplace = True)
for col in df.select_dtypes(include = 'object'):
  df[col]=df[col].fillna(df[col].mode()[0])
print(df.info())
print(df.describe())
X = df.drop(columns=['Investment_Avenues'])
y = df['Investment_Avenues']
le = LabelEncoder()
y = le.fit_transform(y)
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42,stratify=y )

model = LogisticRegression(max_iter=1000,  class_weight= 'balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))
print("Confusion Matrix: ", confusion_matrix(y_test,y_pred))
