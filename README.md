# Venture Funding with Deep Learning

In this challenge, I will use Alphabet Soup a venture capital firm as a company name and data provider. As the story goes, Alphabet Soup’s business team receives many funding applications from startups every day. This team has asked to help them create a model that predicts whether applicants will be successful if funded by Alphabet Soup.

With the CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. I will use machine learning and neural networks to create a binary classifier model that will predict whether an applicant will become a successful business.

## Guide

The steps for this challenge are broken out into the following sections:

* Prepare the data for use on a neural network model.

* Compile and evaluate a binary classification model using a neural network.

* Optimize the neural network model.

### Prepare the Data for Use on a Neural Network Model

I will use Pandas and scikit-learn’s `StandardScaler()`, to preprocess the dataset so that can use it to compile and evaluate the neural network model later.

Steps to follow in this part of the challenge:

1. Read the `applicants_data.csv` file into a Pandas DataFrame. Review the DataFrame, looking for categorical variables that will need to be encoded, as well as columns that could eventually define my features and target variables.

2. Drop the unrelevant columns to the binary classification model from the DataFrame . Two Columns which are  “EIN” (Employer Identification Number) and “NAME” columns will be dropped

3. Encode the dataset’s categorical variables using `OneHotEncoder`, and then place the encoded variables into a new DataFrame.

4. Add the original DataFrame’s numerical variables to the DataFrame containing the encoded variables. I will use `panda concat`


5. Using the preprocessed data, I will create the features (`X`) and target (`y`) datasets. The target dataset will be “IS_SUCCESSFUL”. The remaining columns will define the features dataset.

6. I will use `train_test_split`, to Split the features and target sets into training and testing datasets.

7. Then scale the features data using scikit-learn's `StandardScaler` 

### Compile and Evaluate a Binary Classification Model Using a Neural Network

In this second section of the challenge, I will use TensorFlow to design a binary classification deep neural network model. This model will use the dataset’s features to predict whether an Alphabet Soup&ndash;funded startup will be successful based on the features in the dataset. 

To do so, I will complete the following steps:

1. Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.

2. Compile and fit the model using the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` evaluation metric.

3. Evaluate the model using the test data to determine the model’s loss and accuracy.

4. Save and export the model to an HDF5 file. The file name will be `AlphabetSoup.h5`.

### Optimize the Neural Network Model

In this last section, I will attempts to improve my accuracy

To do so, I will complete the following steps:

1. I will introduce 2 new deep neural network models. With each try, my intention is to improve on my first model’s predictive accuracy.

  My goal is to reach the perfect accuracy that has a value of 1. since accuracy improves as   its value moves closer to 1. To optimize my model for a predictive accuracy as close to 1   as possible, I can use any or all of the following techniques:
    
    * Adjust the input data by dropping different features columns to ensure that no 
      variables or outliers confuse the model. I did this before so I choose not to 
      repeat it since I have other options I can try, such as:
    
    * Adding more neurons (nodes) to a hidden layer.
    
    * Adding more hidden layers.
    
    * Use different activation functions for the hidden layers.
    
    * Add to or reduce the number of epochs in the training regimen.

2. The next step after finishing my model will be to display the accuracy scores achieved by each model, and compare the results.

3. Last will be to Save each of the models as an HDF5 file.


