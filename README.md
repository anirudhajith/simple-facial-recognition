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

## Usage:
Use the command `python3 server.py` to start the server on `localhost:8080`. Feed in the url to an image and hit Submit to view the results.

Facial recognition currently works on 
* Abdul Kalam
* Barack Obama
* Benedict Cumberbatch
* Donald Glover
* Emma Watson
* George Clooney
* Hillary Clinton
* Hugh Laurie
* Indira Gandhi
* Kiera Knightley
* Meryl Streep
* Neil DeGrasse Tyson
* Nelson Mandela
* Rowan Atkinson

To add a new person, create a new folder (titled their name) under `res/targets/` and save one picture of the person inside it. Next, run `python3 generateVectors.py` and restart the server.














