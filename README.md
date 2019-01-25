# Perceptron-classier
A Perceptron classifier (Vanilla and Averaged) using Python to classify hotel reviews as true or fake and positive or negative.
The are two programs:
  # perceplearn.py 
  This program will learn perceptron models (vanilla and averaged) from the training data
  The learning program will be invoked in the following way:
  > python perceplearn.py /path/to/input_file
  The format of the input sentance
  uniqueId 'Fake'/'True' 'Neg'/'Pos' sentance
  Example  : '064BmtQ Fake Neg I was very disappointed with this hotel'
  
  # percepclassify.py 
  This program will use the models to classify new data
  The classification program will be invoked in the following way:
  > python percepclassify.py /path/to/model /path/to/input
  
  The first argument is the path to the model file (vanillamodel.txt or averagedmodel.txt), and the second argument is the path to a file   containing the test data file; the program will read the parameters of a perceptron model from the model file, classify each entry in     the test data, and write the results to a text file called percepoutput.txt.
