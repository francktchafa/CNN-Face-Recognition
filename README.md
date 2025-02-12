This repository showcases two approaches to face recognition using Convolutional Neural Networks (CNN): face recognition with binary classification and face recognition with a triplet loss function.

# Face Recognition with Triplet Loss Function
This approach employs a triplet loss function to train the CNN, enabling it to learn a similarity metric. The model distinguishes between similar and dissimilar face pairs, making it ideal for advanced face recognition tasks.

# Face Recognition with Binary Classification
This approach uses InceptionResNetV2 as the base model, coupled with additional layers for binary classification. The model generates image embeddings that minimize binary cross-entropy loss. Post-training, verification functions determine if a person in the camera image matches an identity in the database. This method is suitable for basic face-recognition tasks.# CNN-Face-Recognition
