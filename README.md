# neuralpackage
# Modelpair

This is a Python package that is designed to compare different machine learning model performance on a dataset. Currently it can only deal with csv file dataset in a certain format

### CURRENT PROGRESS
* Implemented data preprocessing methods for Excel csv file(in certain format)
* Implemented decision tree, KNN and neural network algorithms
* Implemented accuracy comparison methods

### TO INSTALL THE PACKAGE:
``` pip install Modelpair ```

### TO RUN THE PACKAGE FUNCTION:
``` from Modelpair import Compare_class ```
create an instance
``` compare = Compare_class('your csv data file path) ```
Then use the compare function
``` compare.generate_compare() ```


