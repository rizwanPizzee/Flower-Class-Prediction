import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.pipeline import make_pipeline

df = pd.read_csv('iris.csv')

X = df[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
pipeline.fit(X_train, y_train)

pickle.dump(pipeline, open('model.pkl', 'wb'))

