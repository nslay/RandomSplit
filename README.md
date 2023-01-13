# RandomSplit
Fairly partition training/test sets with respect to several counting-based criteria.

# Introduction
In medical image learning tasks, a single image can have several annotated types of pathology present. This contrasts with learning tasks where each example is paired with a single annotation (e.g. class label) where a method like sklearn's `train_test_split` can fairly partition training and test splits. Carelessly partitioning medical images into a training and test set may result in an unbalanced distribution of the pathology. This repository hosts a simple method to fairly partition training and test sets considering counts of pathology present in an image. You can even extend this problem to include a fair partitioning with respect to more counting-based criteria like ISUP grades, tumor volumes, multi-label counts, etc...

# The Method
Suppose you want to sample p% of the images to be your training set so that you have about p% of each kind of pathology/ISUP grade/tumor burden volume/label count/etc...

Given a KxN weight matrix W where N is the number of images and K the number of counting-based criteria, the method proceeds as follows
1. Compute D = inv(diag(W **1**)), Z = I - 1/K. Here **1** is a vector of 1s.
2. Compute SVD: USV^T = ZDW
3. Take Q to be the last N-K+1 column vector of V (the null space of ZDW)
4. Sample **u** ~ N(0,1) where **u** is an N-K+1 dimensional real vector.
5. Compute **x** = Q**u**
6. Take the largest pN components of **x** to be your training set.
7. Optional: A goodness-of-fit can be computed from the indicator vector **x**_train with 1s in place of the largest pN components and 0s elsewhere. Residual = |ZDW**x**_train| where |.| is a norm.

# How it Works
Each column of the weight matrix W represents one image. The rows represent some kind of count of pathology, ISUP grades, tumor volume, label count, etc... The vector W**1** gives a total count/sum of all pathologies, ISUP grades, tumor volumes, etc... The matrix DW gives weighted columns so that DW**1** = **1**. You want to sample p% of the images so that you also have about p% of each pathology, ISUP grade, total tumor volume, etc... In other words, you want an indicator vector **x**_train with pN 1s that gives

DW**x**_train = p**1**

The matrix Z is a rank K-1 orthogonal projection matrix with null space being c**1**. So we can instead solve this problem

ZDW**x**_train = **0**

DW is probably rank K but it does not matter. We'll assume that ZDW is rank K-1. We can compute the null space with SVD

USV^T = ZDW

And take Q to be the last N-K+1 column vectors of V to be the null space. We can sample a random null space vector **x** which will always be a solution to ZDW**x** = **0**

Sample **u** ~ N(0,1) and calculate **x** = Q**u**

Now this solution **x** is *not* an indicator vector! But that's OK. We can always find a linear combination **x**_train = a**x** + b**1** so that **0** <= **x**_train <= **1** and sum(**x**_train) = pN for real numbers a, b. And what do you know, this linear combination remains a solution since DW**1** = **1** and Z**1** = **0**. Thus, picking the top pN components of **x**_train is the same as picking the top pN components of **x**. Hence, you can safely ignore the indicator vector issue and just take **x**_train to have 1s for the top pN components of **x**. The residual |ZDW**x**_train| gives a measurement of how close you are to having the same proportion of everything in your training set (0 being perfect).

# Usage

# Benchmarks

# Remarks

