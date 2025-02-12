"""
Sample code for training a face recognition (FR) model with binary classification.

This code uses the InceptionResNetV2 (base model) coupled with additional layers for face recognition with binary
classification. The model is trained on the dataset using KFold cross-validation.

The weights of the FR model are trained to generate image embeddings, minimizing the binary crossentropy loss.
After training,  one can use the 'verify_id2()' or 'who_is_this2()' functions to determine if a person in the
camera image matches the identity or is included in the database.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras import Model, optimizers, initializers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from pprint import pprint
K.set_image_data_format("channels_last")
tf.random.set_seed(42)


# BUILD MODEL FOR IMAGE EMBEDDING AND BINARY CLASSIFICATION
# Build the base model using the inception network
def encoder_base(img_size=160):
    """
    Builds an encoder model based on the InceptionResNetV2 architecture.

    Arguments:
        img_size -- Size of the input images (default is 160).
        dropout_rate -- Dropout rate for regularization (default is 0.2).
        dimension -- Dimensionality of the output embedding space (default is 128).
        fine_tune_at -- Layer index to start fine-tuning (default is None).

    Returns:
        enc_base -- Face recognition base model
    """

    # Load the base InceptionResNetV2 model pre-trained on ImageNet, excluding the top classification layer
    inception = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                        input_shape=(img_size, img_size, 3),
                                                        weights='imagenet')

    inception.trainable = False
    enc_base = Model(inputs=inception.input, outputs=inception.output)

    return enc_base


base_model = encoder_base()
print(base_model.outputs)  # Output shape=(None, 3, 3, 1536)
print(base_model.outputs)


# ENCODE FACE IMAGES USING BASE ENCODER
# Functions to process images from path
def preprocess_image(img_path, img_size=160):
    """
    Preprocesses an image located at img_path.

    Arguments:
        img_path -- Path to the image file.
        img_size -- Target size for resizing the image (default is 160).

    Returns:
        img -- Preprocessed image tensor.
    """
    target_size = (img_size, img_size)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1] range

    return tf.expand_dims(img, axis=0)


def image_to_encoding(img_path, enc_model, img_size=160):
    """
    Encodes the image located at "img_path" using the specified model.

    Arguments:
        img_path -- Path to the image file.
        enc_model -- The pre-trained model for encoding the image.
        img_size -- Target size for resizing the image (default is 160).

    Returns:
        img_encoding -- Encoded image as a TensorFlow tensor.
    """
    try:
        X = preprocess_image(img_path, img_size)
        img_encoding = enc_model.predict_on_batch(X)
        img_encoding, _ = tf.linalg.normalize(img_encoding, ord=2)
        return img_encoding

    except Exception as e:
        print(f"Error processing image {os.path.basename(img_path)}: {e}")
        return None


# Building a database of the encoded images using the base model
folder_path = os.path.join(os.getcwd(), "images")
valid_extensions = (".png", ".jpg", ".jpeg")
db_base_encodings = {}

for filename in os.listdir(folder_path):
    if filename.endswith(valid_extensions):
        filepath = os.path.join(folder_path, filename)
        name_without_ext = filename.split('.')[0]  # Remove file extension from image name
        db_base_encodings[name_without_ext] = image_to_encoding(filepath, base_model)
        # name_only = filename.split('_')[0]
        # db_encodings[name_only] = image_to_encoding(filepath, base_model)

print(db_base_encodings.keys())
print(len(db_base_encodings.values()))


# PREPARE TRAINING DATASET AND DATABASE OF BASE EMBEDDINGS
def process_encodings(encodings):
    """
    Processes the encodings to create a dataset for training the face recognition model and a database of employees
    for verification/recognition later.

    Arguments:
        encodings -- Python dictionary mapping names of allowed people (strings) to their encodings (matrices).

    Returns:
        pairs -- Differences between encodings of images in the database.
        labels -- 1 if the pair was created using images of the same person, 0 otherwise.
        database -- Python dictionary mapping names of allowed people to their various encodings (matrices).
    """

    pairs = []
    labels = []
    db_keys = list(encodings.keys())
    employees_db = {}

    # Loop over each key in the database
    for i in range(len(db_keys)):
        employee_name = db_keys[i].split('_')[0]  # Extract the name/ID from the key
        anchor = encodings[db_keys[i]]

        # Initialize entry in database if not present
        if employee_name not in employees_db:
            employees_db[employee_name] = []

        # Append the encoding to the database
        employees_db[employee_name].append(encodings[db_keys[i]])

        # Compare with all other keys in the database
        for j in range(i + 1, len(db_keys)):
            current_name = db_keys[j].split('_')[0]  # Extract the name/ID from the comparison key

            if employee_name == current_name:
                positive = encodings[db_keys[j]]
                anc_pos = tf.abs(tf.subtract(anchor, positive))[0]
                pairs.append(anc_pos)
                labels.append(1)  # Label as 1 (same person)
            else:
                negative = encodings[db_keys[j]]
                anc_neg = tf.abs(tf.subtract(anchor, negative))[0]
                pairs.append(anc_neg)
                labels.append(0)  # Label as 0 (not the same people)

    return np.array(pairs), np.array(labels), employees_db  # Convert lists to numpy arrays and return


# Create dataset for training the model
X_data, y_data, my_database = process_encodings(db_base_encodings)
y_data = tf.reshape(tf.convert_to_tensor(y_data), shape=[len(y_data), 1])
print(f"X shape: {X_data.shape}")  # Output shape (253, 3, 3, 1536)
print(f"y shape: {y_data.shape}")  # Output shape: (253, 1)

positive_class_ratio = np.squeeze(sum(np.array(y_data) == 1) / len(y_data))

# Creating class_weights to be used during training since the class is unbalanced
y_data_np = y_data.numpy().flatten()  # Ensure it's 1D
class_weights = compute_class_weight('balanced', classes=np.unique(y_data_np), y=y_data_np)
# class_weights_dict = dict(enumerate(class_weights))
class_weights_dict = {i: round(weight, 2) for i, weight in enumerate(class_weights)}

print("The class weights: ", class_weights_dict)
print(f"Positive samples represent{positive_class_ratio * 100: .2f}% of data.")


def facereco_model(dimension=128, dropout_rate=0.2, input_shape=(3, 3, 1536)):
    """
    Model for face recognition with binary classification

    Arguments:
        input_shape -- Shape of the input base encodings (default is (3, 3, 1536)).
        dropout_rate -- Dropout rate for regularization (default is 0.2).
        dimension -- Dimensionality of the output embedding space (default is 128).
        fine_tune_at -- Layer index to start fine-tuning (default is None).

    Returns:
        fr_model -- Face recognition model ready to generate embeddings.
    """

    base_enc = Input(shape=input_shape)  # This is the output from the base model

    # Add custom layers on top of the base model
    X = GlobalAveragePooling2D()(base_enc)  # Pools the (None, 3, 3, 1536) to (None, 1536)
    X = Dropout(dropout_rate)(X)
    X = Dense(dimension, activation='relu')(X)
    X = BatchNormalization()(X)
    output = Dense(1, activation='sigmoid', kernel_initializer=initializers.glorot_uniform(seed=42))(X)

    # Create the final model
    fr_model = Model(inputs=base_enc, outputs=output)

    return fr_model


# Define KFold cross-validator
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize variables to store cross-validation results
cv_scores, all_history, all_models = [], [], []
best_val_accuracy = 0
best_model_weights = None

# Training with KFold cross-validation
for train_index, val_index in kfold.split(X_data):
    X_train, X_val = tf.gather(X_data, train_index), tf.gather(X_data, val_index)
    y_train, y_val = tf.gather(y_data, train_index), tf.gather(y_data, val_index)

    # Initialize and compile the model inside the loop
    FRmodel = facereco_model()
    FRmodel.compile(optimizer=optimizers.legacy.Adam(learning_rate=0.0001),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=200,
                                   restore_best_weights=True)

    checkpoint = ModelCheckpoint(filepath=os.path.join(os.getcwd(), "models", "best_binary_model.h5"),
                                 monitor='val_loss',
                                 save_best_only=True,
                                 verbose=1)

    # Train the model
    history = FRmodel.fit(X_train, y_train,
                          epochs=2000,
                          batch_size=32,
                          class_weight=class_weights_dict,
                          validation_data=(X_val, y_val),
                          callbacks=[early_stopping, checkpoint])

    all_history.append(history.history)
    all_models.append(FRmodel)

    # Evaluate the model on the validation set and store the results
    scores = FRmodel.evaluate(X_val, y_val, verbose=0)
    cv_scores.append(scores)

    # Track the best model
    if scores[1] > best_val_accuracy:
        best_val_accuracy = scores[1]
        best_model_weights = FRmodel.get_weights()


def plot_metrics(histories, metric):
    """
    Plots the specified metric for training and validation data across multiple folds.

    Parameters:
        histories (list): A list of history objects from Keras training.
        metric (str): The metric to be plotted (e.g., 'loss', 'accuracy').

    Returns:
        None
    """

    for i, hist in enumerate(histories):
        # Plot training metric for the i-th fold
        plt.plot(hist[metric], label=f"Fold {i + 1} Train {metric.title()}")
        # Plot validation metric for the i-th fold
        plt.plot(hist[f"val_{metric}"], label=f"Fold {i + 1} Validation {metric.title()}")
    plt.xlabel("Epochs")
    plt.ylabel(metric.title())
    plt.legend()
    plt.show()


plot_metrics(all_history, "loss")
plot_metrics(all_history, "accuracy")

# Print cross-validation results
print("Cross-validation [loss, accuracy]:", cv_scores)
print("Average cross-validation [loss, accuracy]:", np.mean(cv_scores, axis=0))
print("Best model validation accuracy:", best_val_accuracy)

# Save the best model weights
if best_model_weights is not None:
    best_model = facereco_model()
    best_model.set_weights(best_model_weights)
    best_model.save_weights(os.path.join("models", 'best_trained_binary_model_weights.h5'))


# Just implementing it for fun to compare model checkpoint weights with manual weight selection
def load_and_compare_weights(model, weights_path_one, weights_path_two):
    """
    Loads two sets of weights into the model and compares them to determine if they are identical.

    Arguments:
        model -- The model architecture to be loaded with weights.
        weights_path_one -- Path to the first weights file (weights1.h5).
        weights_path_two -- Path to the second weights file (weights2.h5).

    Returns:
        is_identical -- Boolean indicating if the weights are identical.
    """

    # Load weights into models
    model.load_weights(weights_path_one)
    weights_one = model.get_weights()

    model.load_weights(weights_path_two)
    weights_two = model.get_weights()

    # Compare weights
    is_identical = all(np.array_equal(w1, w2) for w1, w2 in zip(weights_one, weights_two))

    return is_identical


# Compare the two sets of weights
weights1_path = os.path.join("models", 'best_binary_model.h5')
weights2_path = os.path.join("models", 'best_trained_binary_model_weights.h5')
identical = load_and_compare_weights(facereco_model(), weights1_path, weights2_path)

if identical:
    print("The weights are identical.")
else:
    print("The weights are not identical.")


# FACE VERIFICATION: CHECK CAMERA PICTURE MATCHES EMPLOYEE IN DATABASE
def verify_id2(image_path, identity, encoding_model, database, fr_model, threshold=0.7):
    """
    Verifies if the person in the "image_path" image matches the "identity".

    Arguments:
        image_path -- Path to an image file.
        identity -- String, name of the person whose identity you'd like to verify. It must be someone in the database.
        encoding_model -- The base model for initial encoding the images.
        database -- Python dictionary mapping names of allowed people to their various base encodings (matrices)
        fr_model -- Trained face recognition model with binary classification
        threshold -- Probability threshold for validation (default is 0.7).

    Returns:
        proba -- Probability that the "image_path" image is "identity".
        door_open -- True if the door should open, False otherwise.

        Note: Computationally efficient since it only requires computing the probability for the encoding with the
        minimum distance. It's simpler to implement and understand but there are potential risks. There is a risk of
        misclassification if the chosen encoding with the minimum distance is not representative enough.

    """

    # Compute the initial encoding using the base encoder
    anchor = image_to_encoding(image_path, encoding_model)

    employees = list(database.keys())
    current_name = identity
    door_open = False

    pairs = []

    # Loop over each encoding of identity
    if current_name in employees:
        for _, embedding in enumerate(database[current_name]):
            pairs.append(tf.abs(tf.subtract(anchor, embedding))[0])

    else:
        raise ValueError("No such identity in the database")

    # Setting the trained model for inference
    for layer in fr_model.layers:
        layer.trainable = False

    y = fr_model.predict(np.array(pairs), verbose=0)

    if np.any(y > threshold):
        door_open = True
        print("\033[92mIt's " + identity.title() + ", welcome in!" + "\033[39m")

    else:
        print("\033[91m It's not " + str(identity).title() + "; go away or I'll call the police." + "\033[39m")

    return y, door_open


# Testing
pretrained_model = facereco_model()
pretrained_model.load_weights(os.path.join("keras-facenet-h5", 'best_binary_model.h5'))

camera_path = os.path.join(os.getcwd(), "images", "camera", "camera_8.jpg")
p, door_status = verify_id2(image_path=camera_path, identity="franck", database=my_database,
                            encoding_model=base_model, fr_model=pretrained_model)

y_pred = pretrained_model.predict(X_data, verbose=0)
accuracy = accuracy_score(y_true=y_data, y_pred=(y_pred > 0.5).astype(int))
print(f"Accuracy (training dataset): {round(accuracy * 100, 2)}%")


# FACE RECOGNITION: IDENTIFIES THE PERSON IN THE CAMERA IMAGE
def who_is_this2(image_path, encoding_model, database, fr_model, threshold=0.7):
    """
    Determines the identity of the person in the "image_path" image.

    Arguments:
        image_path -- string, Path to an image file.
        encoding_model -- The base model for initial encoding the images.
        database -- Python dictionary mapping names of allowed people to their various base encodings (matrices).
        fr_model -- Trained face recognition model with binary classification.
        threshold -- Probability threshold for validation (default is 0.7).

    Returns:
        identity -- string, the name of the person identified in the image_path.
        max_probability -- Highest probability of math to database keys.
        probabilities -- Python dictionary mapping names to their respective probabilities.
    """
    keys = database.keys()
    probabilities = {}

    # Compute the initial encoding using the base encoder
    for layer in fr_model.layers:
        layer.trainable = False
    img_encoding = image_to_encoding(image_path, encoding_model)

    # Loop over each encoding of identity
    for key in keys:
        probabilities[key] = []  # Initialize the probabilities for each person
        max_probability = 0.0  # Initializing the maximum probability

        for encoding in database[key]:
            enc_difference = tf.abs(tf.subtract(encoding, img_encoding))  # Difference with database image encoding
            y = fr_model.predict(np.array(enc_difference), verbose=0)

            max_probability = max(max_probability, np.squeeze(y))

        probabilities[key].append(max_probability)

    identity = max(probabilities, key=probabilities.get)
    max_probability = probabilities[identity][0]
    probabilities_clean = {name: round(val[0].item(), 3) for name, val in probabilities.items()}

    color_red = "\033[91m"
    color_green = "\033[92m"
    color_reset = "\033[39m"

    if np.max(max_probability) < threshold:
        print(f"{color_red}Not in the database.{color_reset}")
    else:
        print(f"{color_green}\nWelcome in, {identity.title()}.{color_reset}")

    return identity, max_probability, probabilities_clean


# Testing
camera_0_path = os.path.join(os.getcwd(), 'images', 'camera', 'camera_0.jpg')
camera_4_path = os.path.join(os.getcwd(), 'images', 'camera', 'camera_4.jpg')
person_id, proba, all_prob = who_is_this2(image_path=camera_0_path, database=my_database,
                                          encoding_model=base_model, fr_model=pretrained_model)

pprint(all_prob, indent=2)

