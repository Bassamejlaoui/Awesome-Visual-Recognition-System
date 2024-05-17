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


To build this image, run: ```$ docker build -t colemurray/medium-facenet-tutorial -f Dockerfile```

This can take several minutes depending on your hardware# On MBP, ~ 25mins# Image can be pulled from dockerhub below
If you would like to avoid building from source, the image can be pulled from dockerhub using:
***docker pull colemurray/medium-facenet-tutorial***

A GPU supported environment can be found here# nvidia-docker pull colemurray/medium-facenet-tutorial:latest-gpu

## Detect, Crop & Align with Dlib

After creating your environment, you can begin preprocessing.

```$ curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2$ bzip2 -d shape_predictor_68_face_landmarks.dat.bz2```

You’ll use this face landmark predictor to find the location of the inner eyes and bottom lips of a face in an image. These coordinates will be used to center align the image.

This file, sourced from CMU, provides methods for detecting a face in an image, finding facial landmarks, and alignment given these landmarks.

Next, you’ll create a preprocessor for your dataset. This file will read each image into memory, attempt to find the largest face, center align, and write the file to output. If a face cannot be found in the image, logging will be displayed to console with the filename.

As each image can be processed independently, python’s multiprocessing is used to process an image on each available cpu core.

### Getting Results

Now that you’ve created a pipeline, time to get results. As the script supports parallelism, you will see increased performance by running with multiple cores. You’ll need to run the preprocessor in the docker environment to have access to the installed libraries.

Below, you’ll mount your project directory as a volume inside the docker container and run the preprocessing script on your input data. The results will be written to a directory specified with command line arguments.

'''$ docker run -v $PWD:/medium-facenet-tutorial \-e PYTHONPATH=$PYTHONPATH:/medium-facenet-tutorial \-it colemurray/medium-facenet-tutorial python3 /medium-facenet-tutorial/medium_facenet_tutorial/preprocess.py \--input-dir /medium-facenet-tutorial/data \--output-dir /medium-facenet-tutorial/output/intermediate \--crop-dim 180'''

***Code up to this point can be found [Here](https://github.com/ColeMurray/medium-facenet-tutorial/tree/add_alignment)*** 

### Review

Using Dlib, you detected the largest face in an image and aligned the center of the face by the inner eyes and bottom lip. This alignment is a method for standardizing each image for use as feature input.

## Creating Embeddings in Tensorflow

Now that you’ve preprocessed the data, you’ll generate vector embeddings of each identity. These embeddings can then be used as input to a classification, regression or clustering task.

### Download Weights
You’ll use the Inception Resnet V1 as your convolutional neural network. First, create a file to download the weights to the model.

By using pre-trained weights, you are able to apply [transfer learning](https://cs231n.github.io/transfer-learning) to a new dataset, in this tutorial the LFW dataset:


```$ docker run -v $PWD:/medium-facenet-tutorial \-e PYTHONPATH=$PYTHONPATH:/medium-facenet-tutorial \-it colemurray/medium-facenet-tutorial python3 /medium-facenet-tutorial/medium_facenet_tutorial/download_and_extract_model.py \--model-dir /medium-facenet-tutorial/etc```

## Load Embeddings

Below, you’ll utilize Tensorflow’s queue api to load the preprocessed images in parallel. By using queues, images can be loaded in parallel using multi-threading. When using a GPU, this allows image preprocessing to be performed on CPU, while matrix multiplication is performed on GPU.

## Train a Classifier

With the input queue squared away, you’ll move on to creating the embeddings.
First, you’ll load the images from the queue you created. While training, you’ll apply preprocessing to the image. This preprocessing will add random transformations to the image, creating more images to train on.
These images will be fed in a batch size of 128 into the model. This model will return a 128 dimensional embedding for each image, returning a 128 x 128 matrix for each batch.
After these embeddings are created, you’ll use them as feature inputs into a scikit-learn’s SVM classifier to train on each identity. Identities with less than 10 images will be dropped. This parameter is tunable from command-line.
```$ docker run -v $PWD:/medium-facenet-tutorial \-e PYTHONPATH=$PYTHONPATH:/medium-facenet-tutorial \-it colemurray/medium-facenet-tutorial \python3 /medium-facenet-tutorial/medium_facenet_tutorial/train_classifier.py \--input-dir /medium-facenet-tutorial/output/intermediate \--model-path /medium-facenet-tutorial/etc/20170511-185253/20170511-185253.pb \--classifier-path /medium-facenet-tutorial/output/classifier.pkl \--num-threads 16 \--num-epochs 25 \--min-num-images-per-class 10 \--is-train```
'''# ~16 mins to complete on MBP```


### valuating the Results

Now that you’ve trained the classifier, you’ll feed it new images it has not trained on. You’ll remove the is_train flag from the previous command to evaluate your results.

```docker run -v $PWD:/$(basename $PWD) \-e PYTHONPATH=$PYTHONPATH:/medium-facenet-tutorial \-it colemurray/medium-facenet-tutorial \python3 /medium-facenet-tutorial/medium_facenet_tutorial/train_classifier.py \--input-dir /medium-facenet-tutorial/output/intermediate \--model-path /medium-facenet-tutorial/etc/20170511-185253/20170511-185253.pb \--classifier-path /medium-facenet-tutorial/output/classifier.pkl \--num-threads 16 \--num-epochs 5 \--min-num-images-per-class 10```

After inference is on each image is complete, you’ll see results printed to console. At 5 epochs, you’ll see ~85.0% accuracy. Training @ 25 epochs gave results:

![image](https://github.com/mejbass/Awesome-Visual-Recognition-System/assets/130122304/f4b5f916-7e99-4c19-b111-3c9a7a092237)

90.8% @ 25 epochs

# Conclusion
In this tutorial, you learned about the history of machine [learning](https://hackernoon.com/tagged/learning) and how to implement a state of the art pipeline. You utilized docker to manage your library dependencies, offering a consistent environment that is platform agnostic. You used Dlib for preprocessing and Tensorflow + Scikit-learn for training a classifier capable of predicting an identity based on an image.


## Complete Code Here:

[ColeMurray/medium-facenet-tutorial_medium-facenet-tutorial - Facial Recognition Pipeline using Dlib and Tensorflow_github.com](https://github.com/ColeMurray/medium-facenet-tutorial)

**Next Steps:**

- Test on your own dataset!
- Experiment with different hyper parameters
- Train on other labels such as gender or age
- Implement a clustering algorithm to group similar faces
