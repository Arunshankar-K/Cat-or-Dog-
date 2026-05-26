import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# 1) Config
# -----------------------------
IMG_SIZE = (160, 160)     # image size for training
BATCH_SIZE = 32
SEED = 42
EPOCHS = 8

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "cats_vs_dogs.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# 2) Load dataset (TFDS)
# -----------------------------
# TFDS provides the dataset and handles download/caching automatically.
(ds_train_raw, ds_test_raw), ds_info = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:]"],  # 80% train, 20% test
    as_supervised=True,                   # yields (image, label)
    with_info=True
)

num_train = tf.data.experimental.cardinality(ds_train_raw).numpy()
num_test = tf.data.experimental.cardinality(ds_test_raw).numpy()
print(f"Train samples: {num_train}, Test samples: {num_test}")

# We'll carve out a validation set from the training set.
val_fraction = 0.1
num_val = int(num_train * val_fraction)

ds_val_raw = ds_train_raw.take(num_val)
ds_train_raw = ds_train_raw.skip(num_val)

# -----------------------------
# 3) Preprocessing pipeline
# -----------------------------
# a) Resize images to fixed size
# b) Convert to float and normalize to [0, 1]
def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

AUTOTUNE = tf.data.AUTOTUNE

ds_train = (ds_train_raw
            .shuffle(2000, seed=SEED)
            .map(preprocess, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

ds_val = (ds_val_raw
          .map(preprocess, num_parallel_calls=AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(AUTOTUNE))

ds_test = (ds_test_raw
           .map(preprocess, num_parallel_calls=AUTOTUNE)
           .batch(BATCH_SIZE)
           .prefetch(AUTOTUNE))

# -----------------------------
# 4) Data augmentation
# -----------------------------
# This makes the model more robust and helps reduce overfitting.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
])

# -----------------------------
# 5) Build a beginner CNN model
# -----------------------------
# For binary classification, final layer is 1 unit with sigmoid.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(*IMG_SIZE, 3)),
    data_augmentation,

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 6) Train
# -----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True
    ),
]

history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS,
    callbacks=callbacks
)

# -----------------------------
# 7) Evaluate
# -----------------------------
test_loss, test_acc = model.evaluate(ds_test)
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# Extra: classification report + confusion matrix
y_true = []
y_pred = []

for batch_images, batch_labels in ds_test:
    probs = model.predict(batch_images, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    y_true.extend(batch_labels.numpy().tolist())
    y_pred.extend(preds.tolist())

print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=["cat", "dog"]))

print(f"\nSaved best model to: {MODEL_PATH}")