# RandomSplit
Fairly partition training/test sets with respect to several counting-based criteria.

# Introduction
In medical image learning tasks, a single image can have several annotated types of pathology present. This contrasts with learning tasks where each example is paired with a single annotation (e.g. class label) where a method like sklearn's `train_test_split` can fairly partition training and test splits. Carelessly partitioning medical images into a training and test set may result in an unbalanced distribution of the pathology. This repository hosts a simple method to fairly partition training and test sets considering counts of pathology present in an image. You can even extend this problem to include a fair partitioning with respect to more count-based criteria like ISUP grades, tumor volumes, multi-label counts, etc...

# The Method

# How it Works

# Usage
