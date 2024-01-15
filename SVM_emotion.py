import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset


# Load audio files and extract features
def extract_features(audio, sr=16000):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

# Define emotions and corresponding labels
emotions = ['enthusiasm', 'happiness', 'neutral', 'sadness', 'disgust', 'fear', 'anger']
labels = list(range(len(emotions)))

# Load and preprocess data
X_train = []
y_train = []
X_test = []
y_test = []


# Function to load and process data from the dataset
def load_and_process_data(dataset, X, y):
    for label, emotion in enumerate(emotions):
        for example in dataset:
            if example['emotion'] == emotion:
                audio_dict = example['speech']
                audio_array = audio_dict['array']
                # Assuming 'array' is the correct key for audio data
                audio = np.array(audio_array)
                sr = audio_dict['sampling_rate']
                features = extract_features(audio, sr=sr)
                X.append(features)
                y.append(label)


voice_emotion_dataset = load_dataset('Aniemore/resd')
train_dataset = voice_emotion_dataset['train']
test_dataset = voice_emotion_dataset['test']
# Load and preprocess training data
load_and_process_data(train_dataset, X_train, y_train)

# Convert lists to numpy arrays and concatenate features
if X_train:
    X_train = np.vstack(X_train)

# Train a Support Vector Machine (SVM) classifier
clf = SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

# Make predictions on the test set
# (Assuming you want to test on the test dataset, please adjust if necessary)
load_and_process_data(test_dataset, X_test, y_test)

# Check if the test dataset is not empty
if X_test:
    X_test = np.vstack(X_test)
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=emotions))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
else:
    print("Test dataset is empty.")
