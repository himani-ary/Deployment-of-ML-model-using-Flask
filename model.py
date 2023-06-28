import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the CSV file
df = pd.read_csv("Iris.csv")

# Display the first few rows of the dataframe
print(df.head(10))

# Select the independent and dependent variables
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Perform feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Instantiate the random forest classifier model
classifier = RandomForestClassifier()

# Fit the model to the training data
classifier.fit(X_train, y_train)

# Save the trained model as a pickle file
pickle.dump(classifier, open("model.pkl", "wb"))
