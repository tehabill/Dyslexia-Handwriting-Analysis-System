import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16

DATASET_PATH = "data"
MODEL_PATH = "model/dyslexia_cnn_v2.keras"


model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["non dyslexic", "dyslexic"]
)


true_labels = []
predicted_probabilities = []

for images, labels in validation_dataset:

    predictions = model.predict(images, verbose=0)

    true_labels.extend(labels.numpy())
    predicted_probabilities.extend(predictions.flatten())


true_labels = np.array(true_labels)
predicted_probabilities = np.array(predicted_probabilities)


predicted_labels = (predicted_probabilities >= 0.5).astype(int)

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print(f"Accuracy: {accuracy:.2%}")


print("\nClassification Report:")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=[
            "Non-Dyslexic",
            "Dyslexic"
        ],
        zero_division=0
    )
)


cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print("\nConfusion Matrix:")

print(cm)