# Solution to Kaggle's Titanic warm-up (https://www.kaggle.com/c/titanic)
# Uses tensorflow's logistic regression function to predict survival of Titanic
# passengers.
# Accuracy score on test data: 0.77033

# Imports
import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf

# Open datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Returns a DataFrame of features used for validation.
def get_X(dataset):
    X = pd.DataFrame()

    # Transform Pclass feature into boolean features.
    dataset['IsClass1'] = dataset['Pclass'].apply(lambda p: 1 if p == 1 else 0)
    dataset['IsClass2'] = dataset['Pclass'].apply(lambda p: 1 if p == 2 else 0)
    dataset['IsClass3'] = dataset['Pclass'].apply(lambda p: 1 if p == 3 else 0)

    # Transform Parch and SibSp to NumRelatives
    X['NumRelatives'] = dataset['Parch'].add(dataset['SibSp'])

    # Transform Sex into boolean features IsMale, IsFemale
    dataset['IsMale'] = dataset['Sex'].apply(lambda g: 1 if g == 'male' else 0)
    dataset['IsFemale'] = dataset['Sex'].apply(
        lambda g: 1 if g == 'female' else 0)

    # Feature cross for class x sex. Class 3 dudes had it bad.
    for c in ['IsClass1', 'IsClass2', 'IsClass3']:
        for g in ['IsMale', 'IsFemale']:
            X[c + g] = dataset[c].multiply(dataset[g])

    return X

# Returns the target variable (Survived) for the dataset, as a Series
def get_y(dataset):
    return dataset['Survived']


# Trains a tensorflow LinearClassifier
def train_classifier(learning_rate,
                     steps,
                     batch_size,
                     train_X,
                     train_y):

    # Construct feature feature_columns
    feature_cols = ([tf.feature_column.numeric_column(col)
        for col in train_X])

    # Create training input function
    training_input_fn = tf.estimator.inputs.pandas_input_fn(x=train_X,
                                                            y=train_y,
                                                            batch_size=10,
                                                            shuffle=False,
                                                            num_epochs=1)

    # Create optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    # Create classifier
    classifier = tf.estimator.LinearClassifier(feature_columns=feature_cols,
                                               optimizer=optimizer)

    # Store training losses for each iteration
    training_losses = []

    # We'll train in 10 periods so we can monitor how the loss function changes
    # as the model trains
    periods = 10
    steps_per_period = steps / periods

    # Train classifier
    print('Training classifier...')
    for period in range(periods):
        print('Period ' + str(period))

        classifier.train(input_fn=training_input_fn,
                         steps=steps_per_period)

        # Create prediction input function for training set
        prediction_input_fn = tf.estimator.inputs.pandas_input_fn(x=train_X,
                                                                  y=train_y,
                                                                  batch_size=20,
                                                                  shuffle=False,
                                                                  num_epochs=1)

        # Predict classifications for training set
        training_h = classifier.predict(input_fn=prediction_input_fn)
        training_probs = [x['probabilities'] for x in training_h]

        # Compute loss
        training_log_loss = metrics.log_loss(train_y, training_probs)
        print('Loss ' + str(training_log_loss))

        # Record loss in our list
        training_losses.append(training_log_loss)

    # Graph loss over iterations
    # plt.ylabel("Loss")
    # plt.xlabel("Periods")
    # plt.title("Loss vs. Periods")
    # plt.plot(training_losses)
    # plt.show()
    return classifier


train_X = get_X(train_data)
train_X = train_X.dropna()

# Train a linear classifier on our data
linear_classifier = train_classifier(learning_rate=0.01,
                                     steps=1000,
                                     batch_size=20,
                                     train_X=train_X,
                                     train_y=get_y(train_data)[list(
                                                        train_X.index.values)])

# Fill NaNs in test data with 0s
test_data = test_data.fillna(0)

allData = get_X(train_data)
allData['Survived'] = train_data['Survived']

# Run classifier on training data to see how we did
train_input_fn = tf.estimator.inputs.pandas_input_fn(x=get_X(train_data),
                                                     batch_size=20,
                                                     shuffle=False,
                                                     num_epochs=1)
train_h = linear_classifier.predict(input_fn=train_input_fn)
train_probs = [x['probabilities'] for x in train_h]
train_probs = [1 if x[1] >= 0.5 else 0 for x in train_probs]
train_probs = pd.Series(train_probs)

# How did we do on the training data?
num_correct = len(train_probs[train_probs == train_data['Survived']])
num_wrong = len(train_probs[train_probs != train_data['Survived']])
print('# correct on training data: ' + str(num_correct))
print('# wrong on training data: ' + str(num_wrong))
accuracy = float(num_correct) / (num_correct + num_wrong)
print('Accuracy on training data: ' + str(accuracy))

# Run classifier on test data
test_input_fn = tf.estimator.inputs.pandas_input_fn(x=get_X(test_data),
                                                    batch_size=20,
                                                    shuffle=False,
                                                    num_epochs=1)
test_h = linear_classifier.predict(input_fn=test_input_fn)
test_probs = [x['probabilities'] for x in test_h]
test_probs = [1 if x[1] >= 0.5 else 0 for x in test_probs]
test_probs = pd.Series(test_probs)

# Output test data predictions to csv for submission
results = pd.DataFrame()
results['PassengerId'] = test_data['PassengerId']
results['Survived'] = test_probs
results.to_csv('output.csv', index=False)
