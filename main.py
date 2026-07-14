from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()

# Features and target
X = iris.data
y = iris.target

print("Feature Names:")
print(iris.feature_names)

print("\nTarget Names:")
print(iris.target_names)

print("\nDataset Shape:")
print(X.shape)

print("\nFirst 5 Samples:")
print(X[:5])

print("\nFirst 5 Targets:")
print(y[:5])
from sklearn.model_selection import train_test_split

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("\nTraining Samples:", X_train.shape)
print("Testing Samples:", X_test.shape)
from sklearn.preprocessing import StandardScaler

# Create scaler
scaler = StandardScaler()

# Scale training and testing data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFirst Scaled Training Sample:")
print(X_train[0])

from sklearn.neighbors import KNeighborsClassifier

# Create KNN classifier
model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(X_train, y_train)

print("\nKNN Model Trained Successfully!")
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("\nPredicted Classes:")
print(y_pred)

print("\nActual Classes:")
print(y_test)

print("\nModel Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nF1 Score:", f1)

from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Detailed classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

# Visualize confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=iris.target_names
)

plt.title("KNN Confusion Matrix")
# Custom flower prediction
print("\n--- Custom Flower Prediction ---")

sepal_length = float(input("Enter sepal length (cm): "))
sepal_width = float(input("Enter sepal width (cm): "))
petal_length = float(input("Enter petal length (cm): "))
petal_width = float(input("Enter petal width (cm): "))

custom_flower = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

# Scale custom input
custom_flower_scaled = scaler.transform(custom_flower)

# Predict species
prediction = model.predict(custom_flower_scaled)

print("\nPredicted Flower Species:", iris.target_names[prediction[0]])


plt.show()