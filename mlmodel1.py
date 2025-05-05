# Import necessary libraries
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Part 1: Customize the Dataset

# Features: [weight (grams), color, shape]
# Color: 0=green, 1=yellow, 2=red, 3=purple
# Shape: 0=round, 1=long, 2=heart-shaped

# Labels: 0=apple, 1=banana, 2=cherry, 3=papaya, 4=coconut, 5=langka

X = np.array([
    [150, 2, 0],  # apple
    [120, 1, 1],  # banana
    [10, 2, 2],   # cherry
    [130, 2, 0],  # apple
    [110, 1, 1],  # banana
    [5, 2, 2],    # cherry
    [200, 3, 0],  # papaya
    [300, 1, 0],  # coconut
    [25, 2, 2]    # langka
])

y = np.array([
    0, 1, 2, 0, 1, 2, 3, 4, 5
])

# Part 2: Train the Model

# Gumawa tayo ng model gamit ang DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X, y)
print("Model has been trained on the dataset")

# Part 3: Test Your Classifier

# Sample test fruits (di sila kasali sa training data)
test_fruits = np.array([
    [210, 3, 0],  # papaya-like
    [280, 1, 0],  # coconut-like
    [15, 2, 2]    # langka-like
])

# Predict gamit ang trained model
predicted_labels = model.predict(test_fruits)

# Dictionary para i-translate 'yung label into fruit name
fruit_names = {
    0: "apple",
    1: "banana",
    2: "cherry",
    3: "papaya",
    4: "coconut",
    5: "langka"
}

# Display predictions
for i, label in enumerate(predicted_labels):
    print(f"Test fruit #{i+1} is predicted to be: {fruit_names[label]}")

# Part 4: Bonus – Accuracy Evaluation

# isplit natin ang data into training (70%) and testing (30%) using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a new model in training set
model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)

# Predict the testing set
y_pred = model2.predict(X_test)

# Compute and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy in test set is: {accuracy * 100:.2f}%")