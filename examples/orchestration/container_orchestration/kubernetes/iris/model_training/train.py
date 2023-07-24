import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("/data/iris.csv")

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("class", axis=1), df["class"], test_size=0.2
)

# Define and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Save the model
os.makedirs("/data/model", exist_ok=True)
joblib.dump(model, "/data/model/iris_model.pkl")
