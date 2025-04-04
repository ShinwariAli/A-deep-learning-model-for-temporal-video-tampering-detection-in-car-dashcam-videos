#Experiment 1 - Deletion
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Function to extract frames and calculate differences with normalization
def extract_frames_and_diff1(video_path, resize_shape=(250, 200)):
    cap = cv2.VideoCapture(video_path)
    frame_diffs = []
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, resize_shape)

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, frame)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            diff_sum = np.sum(thresh)
            frame_diffs.append(diff_sum)

        prev_frame = frame

    cap.release()

    # Normalize frame differences between 0 and 1
    if frame_diffs:
        frame_diffs = np.array(frame_diffs)
        frame_diffs = (frame_diffs - np.min(frame_diffs)) / (np.max(frame_diffs) - np.min(frame_diffs))

    return frame_diffs

# Function to load dataset and labels
def load_dataset_and_labels(directory, resize_shape=(250, 200)):
    X, y = [], []
    for filename in os.listdir(directory):
        if filename.endswith('.avi') or filename.endswith('.mp4'):
            video_path = os.path.join(directory, filename)
            frames_diff = extract_frames_and_diff1(video_path, resize_shape)
            X.append(frames_diff)

            label = 1 if filename.startswith('forged_video') else 0
            y.append(label)

    return np.array(X, dtype=object), np.array(y)

# Function to build the 1D-CNN model
def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.6))  # Increased dropout to handle overfitting
    model.add(Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Function to preprocess the video for prediction
def preprocess_video(video_path, resize_shape=(250, 200), max_length=None):
    frame_diffs = extract_frames_and_diff1(video_path, resize_shape)
    X_video = np.array(frame_diffs)

    # If max_length is provided, pad or trim the video sequence
    if max_length is not None:
        if len(X_video) < max_length:
            pad_width = max_length - len(X_video)
            X_video = np.pad(X_video, (0, pad_width), mode='constant')
        elif len(X_video) > max_length:
            X_video = X_video[:max_length]

    return X_video

# Function to make predictions on a single video
def predict_single_video(model, video_path, max_length):
    X_video = preprocess_video(video_path, max_length=max_length)
    X_video = np.expand_dims(X_video, axis=0)  # Add batch dimension
    X_video = np.expand_dims(X_video, axis=-1) # Add channel dimension

    # Make predictions
    prediction = model.predict(X_video)

    return prediction

if __name__ == "__main__":
    # Load the dataset
    X, y = load_dataset_and_labels(r'D:\Data sets\D2City\D2City1001_0_delete\Train_Delete')

    # Convert labels to appropriate datatype
    y = np.array(y).astype(np.float32)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

    # Display class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Training set class distribution: {dict(zip(unique, counts))}")

    unique, counts = np.unique(y_test, return_counts=True)
    print(f"Validation set class distribution: {dict(zip(unique, counts))}")

    # Determine the maximum sequence length for padding
    max_length = max(max(len(seq) for seq in X_train), max(len(seq) for seq in X_test))

    # Padding for X_train
    X_train_padded = pad_sequences(X_train, maxlen=max_length, padding='post', dtype='float32')

    # Padding for X_test
    X_test_padded = pad_sequences(X_test, maxlen=max_length, padding='post', dtype='float32')

    # Build the 1D-CNN model
    input_shape = (X_train_padded.shape[1], 1)  # Adjust input shape for 1D-CNN
    model_cnn = build_cnn_model(input_shape)

    # Callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    # Train the model with callbacks
    batch_size = 32
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_padded, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_padded, y_test)).batch(batch_size)
    
    history = model_cnn.fit(train_dataset, epochs=10, validation_data=test_dataset, 
                            callbacks=[early_stopping, reduce_lr])

    # Example usage for prediction
    #video_path = r'D:\Data sets\targetDataset\Forged\Forgery_insertion\vid415.avi'

    # Max sequence length (based on the training data)
    #max_length = 99  # Adjust this to the actual max length used during training

    # Make predictions on the single video
    #prediction = predict_single_video(model_cnn, video_path, max_length)
    #print("Prediction:", prediction)

    # Example thresholding at 0.5
    #if prediction >= 0.5:
     #   print("The video is tampered.")
    #else:
     #   print("The video is original.")



#prediction on a single video
video_path = r'D:\Data sets\BDDA Berkery DeepDrive Attention Dataset_13Frames Randomly Deleted\26.mp4'

# Max sequence length (based on the training data)
max_length = X_train_padded.shape[1]  # Adjust this to the actual max length used during training

# Make predictions on the single video
prediction = predict_single_video(model_cnn, video_path, max_length)
print("Prediction (raw output):", prediction)

# Example thresholding at 0.5 to classify the video as tampered or original
if prediction[0][0] >= 0.5:  # Since 'predict' returns a 2D array, indexing it correctly
    print("The video is tampered.")
else:
    print("The video is original.")

#test on unseen data, precision, recall and f1 score.
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Directory containing unseen test videos
test_directory = r'D:\Data sets\D2City\D2City1001_0_delete\Test_Delete'

# Function to load unseen dataset and labels
def load_unseen_dataset(directory, resize_shape=(250, 200)):
    X_unseen, y_unseen = [], []
    for filename in os.listdir(directory):
        if filename.endswith('.avi') or filename.endswith('.mp4'):
            video_path = os.path.join(directory, filename)
            frames_diff = extract_frames_and_diff1(video_path, resize_shape)
            X_unseen.append(frames_diff)
            label = 1 if filename.startswith('forged_video') else 0
            y_unseen.append(label)
    
    return np.array(X_unseen, dtype=object), np.array(y_unseen)

# Load unseen test dataset
X_unseen, y_unseen = load_unseen_dataset(test_directory)

# Pad the unseen data using the max_length determined from the training data
X_unseen_padded = pad_sequences(X_unseen, maxlen=max_length, padding='post', dtype='float32')

# Make predictions on the unseen data
X_unseen_padded = np.expand_dims(X_unseen_padded, axis=-1)  # Add channel dimension

# Perform predictions on the unseen dataset
predictions = model_cnn.predict(X_unseen_padded)

# Thresholding the predictions at 0.5 to classify
predicted_labels = (predictions >= 0.5).astype(int).flatten()

# Calculate accuracy, precision, recall, and F1 score
accuracy_unseen = accuracy_score(y_unseen, predicted_labels)
precision_unseen = precision_score(y_unseen, predicted_labels)
recall_unseen = recall_score(y_unseen, predicted_labels)
f1_unseen = f1_score(y_unseen, predicted_labels)

# Print evaluation metrics
print(f"Accuracy on unseen test data: {accuracy_unseen * 100:.2f}%")
print(f"Precision: {precision_unseen * 100:.2f}%")
print(f"Recall: {recall_unseen * 100:.2f}%")
print(f"F1 Score: {f1_unseen * 100:.2f}%")

