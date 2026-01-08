#code 1
#NSL-KDD
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Define column names(jahan)

column_names = [ 
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

# 2. Load datasets (jahan) 
train_path = r"KDDTrain+.txt"
test_path = r"KDDTest+.txt"

train_df = pd.read_csv(train_path, names=column_names)
test_df = pd.read_csv(test_path, names=column_names)

# 3. Drop difficulty column
train_df.drop(columns=["difficulty"], inplace=True)
test_df.drop(columns=["difficulty"], inplace=True)

# 4. Binary label: normal = 0, attack = 1
train_df["label"] = train_df["label"].apply(lambda x: 0 if x == "normal" else 1)
test_df["label"] = test_df["label"].apply(lambda x: 0 if x == "normal" else 1)

# 5. Combine for consistent one-hot encoding
combined = pd.concat([train_df, test_df], axis=0)

# 6. One-hot encode
combined = pd.get_dummies(combined, columns=["protocol_type", "service", "flag"])

# 7. Split again
train_data = combined.iloc[:len(train_df), :]
test_data = combined.iloc[len(train_df):, :]

# 8. Separate features and labels
X_train = train_data.drop("label", axis=1)
y_train = train_data["label"]
X_test = test_data.drop("label", axis=1)
y_test = test_data["label"]

# 9. Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 10. Build DNN model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 11. Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 12. Train
history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=128,
                    validation_split=0.2,
                    verbose=1)

# 13. Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}, Loss: {loss:.4f}")

# 14. Predict
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

# 15. Report
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 16. Plot accuracy/loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()


#code 2
#Data Loading & Preprocessing

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt

# Column names
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

# File paths
train_path = r"C:\Users\SOWDA\KDDTrain+.txt"
test_path = r"C:\Users\SOWDA\KDDTest+.txt"

# Load datasets
train_df = pd.read_csv(train_path, names=column_names)
test_df = pd.read_csv(test_path, names=column_names)

# Drop 'difficulty' column
train_df.drop(columns=['difficulty'], inplace=True)
test_df.drop(columns=['difficulty'], inplace=True)

# Convert label to binary: normal=0, attack=1
train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Combine for one-hot encoding
combined = pd.concat([train_df, test_df], axis=0)

# One-hot encoding categorical columns
combined = pd.get_dummies(combined, columns=['protocol_type', 'service', 'flag'])

# Split back
train_data = combined.iloc[:len(train_df), :]
test_data = combined.iloc[len(train_df):, :]

# Separate features and labels
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#  Random Forest 
print("\n--- Random Forest ---")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print(f"Random Forest Accuracy: {rf_model.score(X_test, y_test):.4f}")
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

# SVM
print("\n--- Support Vector Machine ---")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
print(f"SVM Accuracy: {svm_model.score(X_test, y_test):.4f}")
print("SVM Classification Report:\n", classification_report(y_test, svm_pred))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))

#  Deep Neural Network 
print("\n--- Deep Neural Network ---")

# Build model
dnn_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

dnn_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Train model
history = dnn_model.fit(X_train, y_train,
                        epochs=20,
                        batch_size=118,
                        validation_split=0.2,
                        verbose=1)

# Evaluate
loss, accuracy = dnn_model.evaluate(X_test, y_test, verbose=0)
print(f"DNN Test Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

# Predict and report
y_pred_probs = dnn_model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

print("DNN Classification Report:\n", classification_report(y_test, y_pred))
print("DNN Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot training history
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('DNN Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('DNN Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()




#code 3
#intrusion_detection_models.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# Confusion matrix for DNN (example)
cm_dnn = np.array([[9023, 688],
                   [4107, 8726]])

# DNN training history (example)
history = {
    'accuracy': [0.94, 0.98, 0.99, 0.99, 0.99],
    'val_accuracy': [0.99, 0.99, 0.99, 0.99, 0.99],
    'loss': [0.15, 0.04, 0.03, 0.02, 0.02],
    'val_loss': [0.03, 0.02, 0.02, 0.01, 0.01]
}

# Accuracy of the models
models = ['Random Forest', 'SVM', 'DNN']
accuracies = [75.38, 75.72, 78.73] 

# 1. Confusion Matrix Heatmap for DNN
plt.figure(figsize=(4,4))
sns.heatmap(cm_dnn, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - DNN')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 2. Accuracy and Loss Curves for DNN training
epochs = range(1, len(history['accuracy']) + 1)
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.plot(epochs, history['accuracy'], 'bo-', label='Training Accuracy')
plt.plot(epochs, history['val_accuracy'], 'ro-', label='Validation Accuracy')
plt.title('DNN Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, history['loss'], 'bo-', label='Training Loss')
plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
plt.title('DNN Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 3. Bar Chart comparing accuracies of models
plt.figure(figsize=(5,4))
sns.barplot(x=models, y=accuracies, palette='viridis')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
for i, v in enumerate(accuracies):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center')
plt.show()



#code 4
#performance_comparison.py
import matplotlib.pyplot as plt
import numpy as np

models = ['Random Forest', 'SVM', 'Deep Neural Network']
accuracy = [75.38, 75.72, 78.73]  #accuracy value
precision = [ (0.64+0.97)/2, (0.65+0.96)/2, (0.69+0.93)/2 ]  # Normal and  Attack  average precision
recall = [ (0.97+0.59)/2, (0.96+0.59)/2, (0.93+0.68)/2 ]     # average recall
f1_score = [ (0.77+0.73)/2, (0.78+0.73)/2, (0.79+0.79)/2 ]   # average f1-score

x = np.arange(len(models))  
width = 0.2  

fig, ax = plt.subplots(figsize=(6,5))

rects1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy (%)')
rects2 = ax.bar(x - 0.5*width, [a*100 for a in precision], width, label='Precision (%)')
rects3 = ax.bar(x + 0.5*width, [a*100 for a in recall], width, label='Recall (%)')
rects4 = ax.bar(x + 1.5*width, [a*100 for a in f1_score], width, label='F1-Score (%)')

ax.set_ylabel('Performance Metrics (%)')
ax.set_title('Performance Comparison of Models on NSL-KDD Dataset')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

plt.tight_layout()
plt.show()


