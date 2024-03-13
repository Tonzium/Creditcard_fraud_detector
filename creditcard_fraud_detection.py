import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.metrics import Precision, Recall

#print(Counter(data["Class"]))
# Counter({0: 284315, 1: 492}) -> highly imbalanced dataset

# Settings
EPOCHS = 15
BATCH_SIZE = 500
LATENT_DIM = 32
INPUT_DIM = 30 # features

# Current time for naming
TIME_STAMP = int(time.time())

# Predict credit card fraud data
data = pd.read_csv("creditcard.csv")

# Preprocess
# Separate the features from the labels
x = data.drop('Class', axis=1)
y = data['Class']

# Scale the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Convert the scaled features back to a DataFrame
x_scaled_df = pd.DataFrame(x_scaled, columns=x.columns)

# Handle imbalanced dataset
smote = SMOTE()
x_res, y_res = smote.fit_resample(x_scaled_df, y)

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_res), y=y_res)
class_weights_dict = dict(enumerate(class_weights))

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.2, stratify=y_res)

# Model architecture
model = Sequential([
    Dense(256, activation='relu', input_shape=(INPUT_DIM,), kernel_initializer=HeNormal()),
    Dense(128, activation='relu', kernel_initializer=HeNormal()),
    Dense(LATENT_DIM, activation='relu', kernel_initializer=HeNormal()),
    Dense(128, activation='relu', kernel_initializer=HeNormal()),
    Dense(256, activation='relu', kernel_initializer=HeNormal()),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy", Precision(), Recall()])

# Fit and save training history for plotting
history = model.fit(x_train, y_train,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          shuffle=True,
          class_weight=class_weights_dict,
          validation_data=(x_test, y_test))

# Save model
model.save(f"saved_model_{BATCH_SIZE}bz_{EPOCHS}ep_{TIME_STAMP}.h5")

# Use the model to predict the test set
x_pred = model.predict(x_scaled_df)

# Threshold predictions to get binary class labels
threshold = 0.5
y_pred_binary = (x_pred > threshold).astype(int)

# Create a confusion matrix
conf_matrix = confusion_matrix(y, y_pred_binary)

# Heatmap for the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.title('Confusion Matrix')
plt.savefig(f"Credit card fraud {TIME_STAMP}.png")

# Heatmap for the classification report
report = classification_report(y, y_pred_binary, output_dict=True)
report_df = pd.DataFrame(report).transpose()

report_df.drop(columns='support', inplace=True, errors='ignore')

# Heatmap for the classification report
plt.figure(figsize=(8, 6))
sns.heatmap(report_df.iloc[:-1, :].astype(float), annot=True, cmap='Greens', fmt='.2f')  # Exclude the last row for averages
plt.title('Classification Report')
plt.savefig(f"Classification Report {TIME_STAMP}.png")


# Plotting the history

# Plotting the learning curves
plt.figure(figsize=(14, 4))

# Accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Epochs vs. Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Precision
plt.subplot(1, 3, 2)
plt.plot(history.history['precision'], label='Training Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.title('Epochs vs. Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()

# Recall
plt.subplot(1, 3, 3)
plt.plot(history.history['recall'], label='Training Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.title('Epochs vs. Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()

plt.tight_layout()
plt.savefig('Learning Curves.png')