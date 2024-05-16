# Awesome-Visual-Recognition-System
![image](https://github.com/mejbass/Awesome-Visual-Recognition-System/assets/130122304/0048db68-3e00-41db-b890-effc982caf8d)
-----
Facial recognition is a biometric solution that measures unique characteristics about one’s face. Applications available today include flight checkin, tagging friends and family members in photos, and “tailored” advertising.
To perform facial recognition, you’ll need a way to uniquely represent a face. In 1960, Woodrow Bledsoe used a technique involving marking the coordinates of prominent features of a face. Among these features were the location of hairline, eyes and nose.
In 2015, researchers from Google released a paper, FaceNet, which uses a convolutional neural network relying on the image pixels as the features, rather than extracting them manually. It achieved a new record accuracy of 99.63% on the LFW dataset.

**FaceNet:** In the FaceNet paper, a convolutional neural network architecture is proposed. For a loss function, FaceNet uses “triplet loss”. Triplet loss relies on minimizing the distance from positive examples, while maximizing the distance from negative examples.

![image](https://github.com/mejbass/Awesome-Visual-Recognition-System/assets/130122304/541c04b6-5e5b-4921-983d-233aff56529b)
Triplet loss equation
---
![image](https://github.com/mejbass/Awesome-Visual-Recognition-System/assets/130122304/5a41485c-3489-4564-ace9-e69f6eb4007c)
Triplet loss Learning
---
Conceptually, this makes sense. Faces of the same identity should appear closer to each other than faces of another identity.

**Vector Embeddings:** For this tutorial, the important take away from the paper is the idea of representing a face as a 128-dimensional embedding. An **embedding** is the collective name for mapping input features to vectors. In a facial recognition system, these inputs are images containing a subject’s face, mapped to a numerical vector representation.

![image](https://github.com/mejbass/Awesome-Visual-Recognition-System/assets/130122304/0f0013e9-7d2f-410f-84ca-82f8f995d1b2)
Mapping input to embedding [source](https://fr.slideshare.net/BhaskarMitra3/vectorland-brief-notes-from-using-text-embeddings-for-search)
---
Since these vector embeddings are represented in shared vector space, vector distance can be used to calculate the similarity between two vectors. In a facial recognition context, this can vector distance be applied to calculate how similar two faces are. Additionally, these embeddings can be used as feature inputs into a classification, clustering, or regression task.

![image](https://github.com/mejbass/Awesome-Visual-Recognition-System/assets/130122304/897a5594-3930-4ae1-9ee2-ecc56f828f51)
Example of plotting embeddings in a 3D vector space
---

## Preprocessing Data using Dlib and Docker

Project Structure

├── Dockerfile├── etc│ ├── 20170511–185253│ │ ├── 20170511–185253.pb├── data├── medium_facenet_tutorial│ ├── align_dlib.py│ ├── download_and_extract_model.py│ ├── __init__.py│ ├── lfw_input.py│ ├── preprocess.py│ ├── shape_predictor_68_face_landmarks.dat│ └── train_classifier.py├── requirements.txt


**Preparing the Data**
You’ll use the LFW (Labeled Faces in the Wild) dataset as training data. The directory is structured as seen below. You can replace this with your dataset by following the same structure.

Download [lfw dataset$ curl -O](http://vis-www.cs.umass.edu/lfw/lfw.tgz) # 177MB$ tar -xzvf lfw.tgz

Directory Structure# ├── Tyra_Banks# │ ├── Tyra_Banks_0001.jpg# │ └── Tyra_Banks_0002.jpg# ├── Tyron_Garner# │ ├── Tyron_Garner_0001.jpg# │ └── Tyron_Garner_0002.jpg

## Preprocessing

Below, you’ll pre-process the images before passing them into the FaceNet model. Image pre-processing in a facial recognition context typically solves a few problems. These problems range from lighting differences, occlusion, alignment, segmentation. Below, you’ll address segmentation and alignment.
First, you’ll solve the segmentation problem by finding the largest face in an image. This is useful as our training data does not have to be cropped for a face ahead of time.
Second, you’ll solve alignment. In photographs, it is common for a face to not be perfectly center aligned with the image. To standardize input, you’ll apply a transform to center all images based on the location of eyes and bottom lip.

![image](https://github.com/mejbass/Awesome-Visual-Recognition-System/assets/130122304/f0c9f523-2851-431d-ba48-6d9871d44db3)

## Environment Setup

Here, you’ll use docker to install tensorflow, opencv, and Dlib. Dlib provides a library that can be used for facial detection and alignment. These libraries can be a bit difficult to install, so you’ll use Docker for the install.
Docker is a container platform that simplifies deployment. It solves the problem of installing software dependencies onto different server environments. If you are new to docker, you can read more here. To install docker, run curl https://get.docker.com | sh

After installing docker, you’ll create two files. A ```requirements.txt``` for the python dependencies and a ```Dockerfile``` to create your docker environment.


