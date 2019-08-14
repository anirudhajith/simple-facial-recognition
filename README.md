# Simple Facial Recognition

Serves a simple html form where either an http url or the path to a local image file can be provided. Upon submission, MTCNN is used to extract faces from the image and uses FaceNet to generate 128-dimensional embeddings for them.

The embeddings are then compared to a list of known faces in `res/targets/` to label the faces appropriately. The images are finally rendered with boxes around the detected faces and labels corresponding to the names of the recognised faces.

Use these commands to install prerequisites:
```
sudo apt install python3-pip
sudo pip3 install numpy
sudo pip3 install tensorflow
sudo pip3 install keras
sudo pip3 install opencv-python
sudo pip3 install mtcnn
```















