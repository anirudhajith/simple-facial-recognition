# Simple Facial Recognition

Serves a simple html form where either an http url or the path to a local image file can be provided. Upon submission, MTCNN is used to extract faces from the image and uses FaceNet to generate 128-dimensional embeddings for them.

The embeddings are then compared to a list of known faces in `/res/targets/` to label the faces appropriately. The images are finally rendered with boxes around the detected faces and labels corresponding to the names of the recognised faces.















