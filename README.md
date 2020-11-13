# Drift

**Data Drift Detection**

Drift estimator for multiple columns using cluster sampling and weight weights

It is based on an ADWIN (ADaptive WINdowing) model for each column of a dataframe.
ADWIN is an adaptive sliding window algorithm for detecting changes, and keep up-to-date statistics on a data stream. 
ADWIN allows algorithms not adapted for drifting data, be resistant to this phenomenon.

The general idea is to keep statistics from a variable size window while detecting concept drift.

The algorithm will decide the size of the window by cutting the statistics window at different points and analyze the average of some statistics in these two windows. If the absolute value of the difference between the two averages exceed a predefined threshold, the change is detected at that point and all data before that point is discarded.

When training the model, the size of the resulting dataset is saved (if a sample was performed in the training, the sample size determines the dataframe size, see "size" attributes).
The results should be evaluated at the dataframe level in general or per column (and not at the row level).

Always automatically adjusts the size of the input dataframe to the size of the dataset used in training.

- Example dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
- Based on: https://scikit-multiflow.readthedocs.io/en/stable/api/generated/skmultiflow.drift_detection.ADWIN.html#skmultiflow.drift_detection.ADWIN
