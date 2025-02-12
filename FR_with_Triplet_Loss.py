"""
Sample code for training a face recognition (FR) model using the triplet loss function.
This code utilizes the InceptionResNetV2 as the base model, coupled with additional layers to output a 128-D embedding.

Steps and benefits:
- Generate Embeddings: The FR model is fine-tuned to generate image embeddings that minimize the triplet loss.
- Post-training Usage: After training, embeddings can be generated for new images.
- Functionality: Use the 'verify_id()' or 'who_is_this()' functions to determine if a person in the camera image
  matches the identity or exists in the database.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
import pprint
import os
import itertools

tf.random.set_seed(42)


# BUILD MODEL FOR IMAGE EMBEDDING
# Build the base model using the inception network
def encoder(img_size=160, dropout_rate=0.2, dimension=128, fine_tune_at=None):
    """
    Builds an encoder model based on the InceptionResNetV2 architecture.

    Arguments:
        img_size -- Size of the input images (default is 160).
        dropout_rate -- Dropout rate for regularization (default is 0.2).
        dimension -- Dimensionality of the output embedding space (default is 128).
        fine_tune_at -- Layer index to start fine-tuning (default is 700).

    Returns:
        enc_model -- Encoder model ready to generate embeddings.
    """

    # Load the base InceptionResNetV2 model pre-trained on ImageNet, excluding the top classification layer
    base_model = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                         input_shape=(img_size, img_size, 3),
                                                         weights='imagenet')
    # Freeze all layers in the base model to prevent them from being trained initially
    base_model.trainable = False

    # Optional: Unfreeze some layers for fine-tuning if fine_tune_at is specified
    if fine_tune_at is not None:
        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True

    # Add custom layers on top of the base model
    X = base_model.output
    X = GlobalAveragePooling2D()(X)
    X = Dropout(dropout_rate)(X)
    X = Dense(dimension, activation='relu')(X)  # Default dimension is 128
    X = BatchNormalization()(X)

    # Create the final model
    enc_model = Model(inputs=base_model.input, outputs=X)

    return enc_model


model = encoder(fine_tune_at=700)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(model.summary())

num_layers = len(model.layers)  # Check the number of layers
print(f"Number of layers in the model: {num_layers}")  # 784 layers
print(model.inputs)  # Input shape=(None, 160, 160, 3) as expected
print(model.outputs)  # Output shape=(None, 128) as expected


# Custom class for image embedding
class ImageEmbedding(layers.Layer):
    """
    Custom layer for generating image embeddings using a pre-trained model.
    Extends the TensorFlow/Keras layers.Layer class to integrate seamlessly with the Keras API.
    """

    # Using layers.Layer allows for creating complex, custom behaviors in a way that integrates seamlessly with the
    # rest of the TensorFlow/Keras ecosystem.

    def __init__(self, embedding_model, img_size=160, **kwargs):
        super().__init__(**kwargs)
        self.embedding_model = embedding_model
        self.image_size = img_size

    def call(self, inputs):
        # Generate the embeddings
        X = inputs
        X = self.embedding_model(X)
        img_embedding, _ = tf.linalg.normalize(X, ord=2)
        return img_embedding


input_shape = (160, 160, 3)
anchor = Input(name="anchor", shape=input_shape)
positive = Input(name="positive", shape=input_shape)
negative = Input(name="negative", shape=input_shape)

# Generate the embeddings
embedding_layer = ImageEmbedding(embedding_model=model)
embedded_anchor = embedding_layer(anchor)
embedded_positive = embedding_layer(positive)
embedded_negative = embedding_layer(negative)


# Concatenate the embeddings
apn_embeddings = tf.concat([embedded_anchor, embedded_positive, embedded_negative], axis=1)

# Define the model
triplet_model = Model(inputs=[anchor, positive, negative],
                      outputs=apn_embeddings)


# Define the triplet loss function
def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss for face recognition model

    Arguments:
        y_true -- True labels. Not needed but added to comply with Keras requirements for all custom loss functions.
        y_pred -- Python list containing three objects:
                anc -- Embeddings for the anchor images, of shape (None, 128)
                pos -- Embeddings for the positive images, of shape (None, 128)
                neg -- Embeddings for the negative images, of shape (None, 128)

    Returns:
        loss -- real number, value of the loss
    """

    anc, pos, neg = tf.split(y_pred, num_or_size_splits=3, axis=1)  # Split into anchor, positive, and negative
    pos_distance = tf.reduce_sum(tf.square(tf.subtract(anc, pos)), axis=-1)
    neg_distance = tf.reduce_sum(tf.square(tf.subtract(anc, neg)), axis=-1)
    Z = tf.subtract(tf.add(pos_distance, alpha), neg_distance)
    loss = tf.reduce_sum(tf.maximum(Z, 0))

    return loss


# LOAD DATA AND TRAIN THE MODEL
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
        X = enc_model.predict_on_batch(X)
        img_encoding, _ = tf.linalg.normalize(X, ord=2)
        return img_encoding

    except Exception as e:
        print(f"Error processing image {os.path.basename(img_path)}: {e}")
        return None


def create_dataset(img_dir, valid_extensions=(".png", ".jpg", ".jpeg")):
    """
    Generates triplets for training a face recognition model. While triplets are generated randomly here, it is
    important to select challenging triplets to ensure effective training. Easy examples can render gradient descent
    ineffective, as the neural network may classify them correctly without much effort.

    Arguments:
        img_dir -- path of the directory containing the images.
        valid_extensions -- tuple of valid image file extensions

    Returns:
        triplets -- List of tuples containing (anchor, positive, negative) image paths.
        employees -- Dictionary mapping identities (names) to lists of image file paths
    """

    # Construct the full path for all images
    image_paths_list = [os.path.join(img_dir, img) for img in os.listdir(img_dir) if
                        img.endswith(valid_extensions)]

    # Initialize a dictionary to group images by person_id
    employees = {}

    for image in image_paths_list:
        # Extract the base name of the image file (e.g., 'person1_1.jpg')
        img_name = os.path.basename(image).split('.')[0]  # Removes the file extension
        # Extract the person_id from the image name (e.g., 'person1')
        person_id = img_name.split('_')[0]

        # Initialize the list for this person_id if not already present
        if person_id not in employees:
            employees[person_id] = []
        # Append the image path to the list of this person_id
        employees[person_id].append(image)

    triplets = []

    for person_id, person_images in employees.items():
        # Generate all combinations of anchor and positive images for the same person
        anchor_positive_combinations = list(itertools.combinations(person_images, 2))

        # Find negative images (images from other persons)
        negative_images = [img for pid, imgs in employees.items() if pid != person_id for img in imgs]

        # Create triplets by combining each anchor-positive pair with all negative images
        for anc, pos in anchor_positive_combinations:
            for neg in negative_images:
                triplets.append((anc, pos, neg))

    return triplets, employees


images_dir = os.path.join(os.getcwd(), "images")  # Directory containing images
trios, staff = create_dataset(images_dir)

# Verifying everything works as planned
for trio in trios[:5]:  # Print the first 5 triplets as an example
    print(trio)


def process_triplets(anchor_paths, positive_paths, negative_paths, img_size=160):
    """
    Processes image paths into preprocessed image tensors for anchors, positives, and negatives.

    Arguments:
        anchor_paths -- List of anchor image file paths.
        positive_paths -- List of positive image file paths.
        negative_paths -- List of negative image file paths.
        img_size -- Desired size for resizing the images.

    Returns:
        anchors, positives, negatives -- Preprocessed TensorFlow tensors.
    """
    A = [preprocess_image(path, img_size)[0] for path in anchor_paths]    # anchors
    P = [preprocess_image(path, img_size)[0] for path in positive_paths]  # positives
    N = [preprocess_image(path, img_size)[0] for path in negative_paths]  # negatives

    return tf.stack(A), tf.stack(P), tf.stack(N)


# Split triplets paths into anchors, positives, and negatives paths
anc_paths, pos_paths, neg_paths = [], [], []

for trio in trios:
    a, p, n = trio[0], trio[1], trio[2]
    anc_paths.append(a)
    pos_paths.append(p)
    neg_paths.append(n)

anchors, positives, negatives = process_triplets(anc_paths, pos_paths, neg_paths)

# Let's create a placeholder y_dummy for compatibility with Keras fit API.
# Although required, y_dummy is not used in triplet loss calculation.
m = anchors.shape[0]         # Number of samples in the dataset
n = apn_embeddings.shape[1]  # Concatenated output shape (3 * dimension)
print(f"Shape of dummy variable, y_dummy: ({m},{n})")

y_dummy = tf.zeros([m, n])

# Compile the model
triplet_model.compile(optimizer=Adam(learning_rate=0.0001), loss=triplet_loss)

# Training the model
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
history = triplet_model.fit([anchors, positives, negatives], y_dummy,
                            epochs=30, batch_size=32, callbacks=early_stopping)

plt.plot(history.history['loss'], label='cost')
plt.xlabel('epochs')
plt.ylabel('Cost')
plt.show()


# Save the trained model
model.save_weights(os.path.join(os.getcwd(), "models", "best_trained_encoder.h5"))  # Saving encoder weights


# A Few functions before testing
def create_database(img_paths, enc_model):
    """
    Creates a database of image encodings using the specified model and normalizes them.

    Arguments:
        img_paths -- Dictionary mapping identities (names) to lists of image file paths.
        enc_model -- The pre-trained model for encoding the images.

    Returns:
        database -- Dictionary mapping identities to their normalized encodings.
    """
    database = {}
    for identity, paths in img_paths.items():
        encodings = []
        for path in paths:
            img_encoding = image_to_encoding(path, enc_model)
            if img_encoding is not None:
                encodings.append(img_encoding)

        if encodings:
            database[identity] = tf.stack(encodings)  # Stack encodings for the same identity

    return database


final_model = encoder()
final_model.load_weights(os.path.join(os.getcwd(), "models", "best_trained_encoder.h5"))

# Set model for inference
final_model.trainable = False

my_database = create_database(staff, final_model)


def calculate_max_distances(database):
    """
    Calculate the maximum distance between encodings for the same identity in the database.

    Arguments:
        database -- Dictionary mapping identities to their normalized encodings.

    Returns:
        max_distances -- Array of maximum distances for each identity (sorted in ascending order).
    """
    max_distances = []

    for identity, encodings in database.items():
        num_encodings = encodings.shape[0]

        # Skip if only one encoding
        if num_encodings < 2:
            max_distances.append(0.0)
            continue

        max_distance = 0

        # Calculate pairwise distances
        for i in range(num_encodings):
            for j in range(i + 1, num_encodings):
                distance = np.linalg.norm(encodings[i] - encodings[j])
                if distance > max_distance:
                    max_distance = distance

        max_distances.append(max_distance)

    max_distances = np.array(max_distances)
    print("Min. distance between embeddings:", np.min(max_distances))
    print("Max. distance between embeddings:", np.max(max_distances))

    return np.sort(max_distances)


# Calculate F1 scores to find the best threshold
def find_threshold(database, thresholds):
    """
    Finds the best threshold based on the F1 score for the given database of encodings.

    Arguments:
        database -- Dictionary mapping identities to their normalized encodings.
        thresholds -- List or array of threshold values.

    Returns:
        best_threshold -- Threshold that yields the highest F1 score.
        best_f1 -- The highest F1 score obtained.
        f1_values -- List of F1 scores for each threshold.
    """

    best_f1 = 0
    best_threshold = 0
    f1_values = []
    step_size = (max(thresholds) - min(thresholds)) / 1000

    for threshold in np.arange(min(thresholds), max(thresholds), step_size):
        y_true = []
        y_pred = []

        # Iterate over each identity and their encodings in the database
        for identity, encodings in database.items():
            # Iterate over each encoding as an anchor
            for i in range(encodings.shape[0]):
                current_anchor = encodings[i]
                # Compare with all other encodings of the same identity (positives)
                for j in range(encodings.shape[0]):
                    if i == j:
                        continue  # Skip comparison with itself
                    current_positive = encodings[j]
                    pos_dist = np.linalg.norm(current_anchor - current_positive, ord=2)  # Calculate positive distance
                    y_true.append(1)  # True label for positive pairs
                    y_pred.append(1 if pos_dist < threshold else 0)  # Predicted label based on threshold

                # Compare with all encodings of different identities (negatives)
                for other_identity, neg_encodings in database.items():
                    if other_identity == identity:
                        continue  # Skip the same identity
                    for k in range(neg_encodings.shape[0]):
                        current_negative = neg_encodings[k]
                        neg_dist = np.linalg.norm(current_anchor - current_negative, ord=2)  # Calculate negative distance
                        y_true.append(0)  # True label for negative pairs
                        y_pred.append(1 if neg_dist < threshold else 0)  # Predicted label based on threshold

        # Calculate the F1 score for the current threshold
        f1 = f1_score(y_true, y_pred)
        f1_values.append(f1)

        # Update the best F1 score and threshold
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1, f1_values


max_dists = calculate_max_distances(my_database)
best_t, best_f1_score, f1_vals = find_threshold(my_database, max_dists)


def verify_id(image_path, identity, database, pretrained_model, threshold=0.2):
    """
    Function that verifies if the person in the "image_path" image matches the "identity".

    Arguments:
        image_path -- Path to the image file.
        identity -- String, name of the person whose identity you'd like to verify.
        database -- Python dictionary mapping names of allowed people to their encodings.
        pretrained_model -- The pre-trained model instance used for embedding the images.
        threshold -- Distance threshold for verification (default is 0.2).

    Returns:
        distance -- Distance between the image at image_path and the stored image of "identity" in the database.
        door_open -- True if the identity is verified and the door should open, False otherwise.
    """

    # Set pre-trained model for inference
    for layer in pretrained_model.layers:
        layer.trainable = False


    try:
        # Compute the image encoding using the pre-trained model
        img_encoding = image_to_encoding(image_path, pretrained_model)

        if img_encoding is None:
            raise ValueError("Failed to compute image embedding.")

        # Initialize variables to track the minimum distance
        min_dist = float('inf')
        person_id = None

        # Ensure the provided identity exists in the database
        if identity in database:
            # Compute the distance between the img. embeddings and stored embeddings for the identity and update dist.
            for embedding in database[identity]:
                dist = np.linalg.norm(img_encoding - embedding, ord=2)

                if dist < min_dist:
                    min_dist = dist
                    person_id = identity
        else:
            raise ValueError(f"Identity '{identity}' not found in the database.")

        distance = min_dist
        # Determine if the identity is verified based on the distance threshold
        color_green = "\033[92m"
        color_red = "\033[91m"
        color_reset = "\033[39m"

        if distance < threshold:
            print(f"{color_green}It's {identity.title()}, welcome in!{color_reset}")
            door_open = True
        else:
            print(f"{color_red}It's not {identity.title()}, go away or I'll call the police.{color_reset}")
            door_open = False

    except Exception as e:
        # Print an error message if an exception occurs during processing
        print(f"Processing error: {e}")
        distance = None
        door_open = False

    return distance, door_open


def who_is_this(image_path, database, pretrained_model, threshold=0.2):
    """
    Determines the identity of the person in the image at "image_path".

    Arguments:
        image_path -- Path to an image.
        database -- Dictionary mapping names of allowed people to their encodings.
        pretrained_model -- The pre-trained model instance used for embedding the images.
        threshold -- Distance threshold for verification (default is 0.2).

    Returns:
        identified -- The identified person in the database.
        min_distance -- The minimum distance to the encoding in the database.
        all_distances -- Dictionary mapping names of possible people with their distances to the database encodings.
    """

    # Set pre-trained model for inference
    for layer in pretrained_model.layers:
        layer.trainable = False

    # Encode the input image
    img_encoding = image_to_encoding(image_path, pretrained_model)
    all_distances = {}

    # Iterate over each person in the database and find the minimum distance
    for identity, encodings in database.items():
        differences = [np.linalg.norm(img_encoding - encoding) for encoding in encodings]
        min_distance = min(differences)
        all_distances[identity] = min_distance

    # Find the identity with the smallest distance
    identified = min(all_distances, key=all_distances.get)
    min_distance = all_distances[identified]

    # Check if the minimum distance is below the threshold
    color_green = "\033[92m"
    color_red = "\033[91m"
    color_reset = "\033[39m"

    if min_distance < threshold:
        print(f"{color_green}It's {identified.title()}, welcome in!{color_reset}")

    else:
        identified = None
        print(f"{color_red}Not in the database.{color_reset}")

    return identified, min_distance, all_distances


print(f"The best threshold: {best_t} \nThe best f1 score: {best_f1_score}")

selected_threshold = np.round(best_t, decimals=3)
camera_pic1 = os.path.join(os.getcwd(), "images", "camera", "camera_8.jpg")

dist, door_status = verify_id(camera_pic1, "franck",
                              database=my_database,
                              pretrained_model=final_model,
                              threshold=selected_threshold)


camera_pic2 = os.path.join(os.getcwd(), "images", "camera", "camera_3.jpg")
the_id, the_dist, all_dist = who_is_this(camera_pic2,
                                         database=my_database,
                                         pretrained_model=final_model,
                                         threshold=selected_threshold)

pp.pprint(all_dist)
