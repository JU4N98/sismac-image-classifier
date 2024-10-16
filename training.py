""" 
Parameters of the model: 
* number of layers
* size of the kernel/filter of the ith layer
* number of kernels of the ith layer
* activation function
* usage of pooling layers and kind of them
* optimizer
* loss
* metrics

Models to try:
* model #1: use CNN using the images as input and:
    * first classify images among "sin defectos" and "con defectos"
    * finally classify images belonging to "con defectos" among "sobrecarga en 1 fase", 
    "sobrecarga en 2 fase" and "sobrecarga en 3 fase"
* model #2: use MLP or CNNs using a histogram of the image as input and try to do 
something similar to the previous case.

As an alternative it's also possible to make the whole classification process in the same
neural network.
"""