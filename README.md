# RandomSplit
Fairly partition training/test sets with respect to several counting-based criteria. Fairly partition a data set for F-fold cross validation.

# Introduction
In medical image learning tasks, a single image can have several annotated types of pathology present. This contrasts with learning tasks where each example is paired with a single annotation (e.g. class label) where a method like sklearn's `train_test_split` can fairly partition training and test splits. Carelessly partitioning medical images into a training and test set may result in an unbalanced distribution of the pathology. This repository hosts a simple method to fairly partition training and test sets considering counts of pathology present in an image. You can even extend this problem to include a fair partitioning with respect to more counting-based criteria like ISUP grades, tumor volumes, multi-label counts, etc...

# The Method
Suppose you want to sample p% of the images to be your training set so that you have about p% of each kind of pathology/ISUP grade/tumor burden volume/label count/etc...

Given a KxN weight matrix W *with non-zero rows* where N is the number of images and K the number of counting-based criteria, the method proceeds as follows
1. Compute D = inv(diag(W**1**)), Z = I - 1/K. Here **1** is a vector of 1s.
2. Compute SVD: USV^T = ZDW
3. Take Q to be the last N-K+1 column vector of V (the null space of ZDW)
4. Sample **u** ~ N(0,1) where **u** is an N-K+1 dimensional real vector.
5. Compute **x** = Q**u**
6. Take the largest pN components of **x** to be your training set.
7. Optional: A goodness-of-fit can be computed from the indicator vector **x**_train with 1s in place of the largest pN components and 0s elsewhere. Residual = |ZDW**x**_train| where |.| is a norm.

**NOTE**: A weight matrix W with some zero rows can be modified into a new weight matrix W' with non-zero rows by simply discarding the zero rows of W. The script in this repository automatically performs this check and modification.

**NOTE**: Columns should ideally be non-zero too. A zero column would seem to suggest no annotation is associated with some instance. If you somehow find yourself in this strange situation, add a *self count* row where each instance counts as 1. See the [Benchmarks section](#benchmarks) below for an example. 

# How it Works
Each column of the weight matrix W represents one image. The rows represent some kind of count of pathology, ISUP grades, tumor volume, label count, etc... The vector W**1** gives a total count/sum of all pathologies, ISUP grades, tumor volumes, etc... The matrix DW gives weighted columns so that DW**1** = **1**. You want to sample p% of the images so that you also have about p% of each pathology, ISUP grade, total tumor volume, etc... In other words, you want an indicator vector **x**_train with pN 1s that gives

DW**x**_train = p**1**

The matrix Z is a rank K-1 orthogonal projection matrix with null space being c**1**. So we can instead solve this problem

ZDW**x**_train = **0**

DW is probably rank K but it does not matter. We'll assume that ZDW is rank K-1. We can compute the null space with SVD

USV^T = ZDW

And take Q to be the last N-K+1 column vectors of V to be the null space. We can sample a random null space vector **x** which will always be a solution to ZDW**x** = **0**

Sample **u** ~ N(0,1) and calculate **x** = Q**u**

Now this solution **x** is *not* an indicator vector! But that's OK. We can always find a linear combination **x**_train = a**x** + b**1** for a *soft* form of an indicator vector with **0** <= **x**_train <= **1** and sum(**x**_train) = pN for real numbers a, b. And what do you know, this linear combination remains a solution since DW**1** = **1** and Z**1** = **0**. Thus, picking the top pN components of **x**_train is the same as picking the top pN components of **x**. Hence, you can safely ignore the indicator vector issue and just take **x**_train to have 1s for the top pN components of **x**. The residual |ZDW**x**_train| gives a measurement of how close you are to having the same proportion of everything in your training set (0 being perfect).

## And for Cross Validation?
We can partition the F folds in a balanced way using the sorted **x** values generated by SVD. The sorted indices are partitioned into F evenly sized pieces. Each piece forms a validation set indicator vector for one experiment and its complement is the corresponding training set. To see this, let X be the sorted indices of **x**. The f'th validation set in F-fold cross validation is then

val_f = X_{floor(f*N/F) <= n < floor((f+1)*N/F)} where f = 0, 1, 2, ..., F-1

And the training set is its complement

training_f = val_f^c = X_{0 <= n < floor(f*N/F)} union X_{floor((f+1)*N/F) <= n < N}

Because X represents sorted indices, this partitioning of X can be thought of in terms of disjoint X_top, X_mid and X_bottom largest **x** values. In this case, the f'th fold can be thought of as below

val_f = X_mid

training_f = X_top union X_bottom

From random splitting, we saw that the top p% and bottom (1-p)% of sorted values gives DW**x**_train ~ p**1** and DW**x**_test ~ (1-p)**1**. So if we divide this into the top p%, middle q% and bottom r% with p+q+r = 1, this gives DW**x**_mid ~ q**1** since DW(**x**_top + **x**_mid + **x**_bottom) = DW**1** = **1**. Although we do not actually need to concern ourselves with the values of p, q and r. Conventional cross validation folds of such X remain theoretically balanced.

# Random Split Usage
Form the weight matrix `W` as a KxN `numpy` array. Be sure to create a list or dictionary `column_map` that maps column indices of `W` to image/patient identifiers. The `column_map` may map to single elements or lists of elements. The latter is useful if your columns represent patients but patients were imaged multiple times. Then `column_map` might map to a list of scans for a patient. This prevents contamination of different scans from the same patient across training, validation and test sets. Then just call `MakeRandomSplit` as follows to create a fair (p,q,r)% training, testing and validation set (e.g. p=0.65, q=0.25 and r=0.1 for 65/25/10 training, testing, validation split).
```python
from RandomSplit import MakeRandomSplit

p_train=0.65
p_test=0.25
train_list, test_list, val_list, res_train, res_test = MakeRandomSplit(W, p_train, p_test, column_map, tries=10)
```
The resulting lists can then be directly used or saved to file for training. The `res_train` and `res_test` returns are the goodness-of-fit residuals for the two splitting operations.

You may optionally pass in integer sizes in place of `p_train` and `p_test`. For example if N=200, this would give similar results as the previous example
```python
from RandomSplit import MakeRandomSplit

N_train=130
N_test=50
train_list, test_list, val_list, res_train, res_test = MakeRandomSplit(W, N_train, N_test, column_map)
```

**NOTE**: If your column map maps to lists of identifiers, it would be a good idea to have *self count* row where each column is the length of the list.

# Cross Validation Usage
Form the weight matrix `W` as a KxN `numpy` array. Be sure to create a list or dictionary `column_map` that maps column indices of `W` to image/patient identifiers. The `column_map` may map to single elements or lists of elements. The latter is useful if your columns represent patients but patients were imaged multiple times. Then `column_map` might map to a list of scans for a patient. This prevents contamination of different scans from the same patient across training, validation and test sets. Then just call `MakeBalancedCrossValidation` as follows to create a single fair p% test list (optional) and then balanced `F`-fold cross validation lists from the remainder of the data set.
```python
from RandomSplit import MakeBalancedCrossValidation

F=5
p_test=0.25
training_lists, validation_lists, test_list, res, res_test = MakeBalancedCrossValidation(W, F, column_map, testing_size=p_test, tries=10)
```
In this example, there will be 5 training and validation lists and a single test list comprised of 25% of the total data set. The `res` variable will hold a list of goodness-of-fits for each fold while `res_test` will be the goodness-of-fit for the test list partition (if any). You may choose not to create a separate test set and set `p_test=0.0`.

**NOTE**: By default, the best cross validation partitioning is determined by the maximum residual over the individual folds. You may change what is considered *best* by setting the `aggregator` argument in `MakeBalancedCrossValidation` (by default `aggregator=np.max`).

You may optionally pass in an integer size in place of `p_test`. For example if N=200, this would give similar results as the previous example
```python
from RandomSplit import MakeBalancedCrossValidation

N_test=50
training_lists, validation_lists, test_list, res, res_test = MakeBalancedCrossValidation(W, F, column_map, testing_size=N_test, tries=10)
```

**NOTE**: If your column map maps to lists of identifiers, it would be a good idea to have *self count* row where each column is the length of the list.

# Random Split Indicator Vector Usage
Form the weight matrix `W` as a KxN `numpy` array. Be sure to map columns of `W` to your images in your data set somehow. Then just call `RandomSplit` as follows to create a fair p% training set (e.g. p=0.75 for 75/25 split)
```python
from RandomSplit import RandomSplit

p=0.75
xtrain, residual = RandomSplit(W, p)
xtest = 1-xtrain
```

To get better random splits, you may specify `tries=#` as an argument to `RandomSplit`. For example, 
```python
xtrain, residual = RandomSplit(W, p, tries=10)
```
will compute 10 p% splits and return the one with the lowest residual (10 is the default).

Instead of providing a ratio 0 < p < 1, you may provide an integer for the training set size. For example,
```python
xtrain, residual = RandomSplit(W, 100)
```
will pick a training set to have 100 examples.

# Cross Validation Indicator Vector Usage
Form the weight matrix `W` as a KxN `numpy` array. Be sure to map columns of `W` to your images in your data set somehow. Then just call `BalancedCrossValidation` as follows to create a balanced `F`-fold cross validation indicator vectors for training.
```python
from RandomSplit import BalancedCrossValidation

F=5
folds, residuals = BalancedCrossValidation(W, F)

for f in range(F):
  xtrain = folds[f]
  xtest = 1 - xtrain
  fold_res = residuals[f]
  # Do something here...
```

To get better folds, you may specify `tries=#` as an argument to `BalancedCrossValidation`. For example, 
```python
folds, residuals = BalancedCrossValidation(W, F, tries=10)
```
will compute 10 `F`-fold partitions and return the one with the *lowest* residual (10 is the default). Here, *lowest* means a `F`-fold partitioning that has the lowest **maximum** residual over the folds.

Instead of using the maximum, you can set `aggregator` in `BalancedCrossValidation` (by default `aggregator=np.max`).
```python
folds, residuals = BalancedCrossValidation(W, F, aggregator=np.mean, tries=10)
```
This will instead average the `F` fold residuals for comparison purposes.

# Benchmarks
As a simple test, I performed 100,000 runs of 50/50 splits of N=200 cases with K=11 criteria. The baseline method is to just randomly shuffle {0,1,2,...,199} and take the first 50% of the indices as the training set. The indicator vector **x**_train is 1 for each of those indices. The weight matrix W is 11x200 and is randomly constructed for each run as follows
* W[0, :] = 1 -- Each column counts as 1 instance. You don't necessarily need this row, but this avoids a corner case where somehow a p% split gives some other q% of each criteria. This row encourages q=p.
* W[1-6 and 8-10, :] ~ U[1,10)
* W[7, :] = 0 -- This tests automatic removal of rows where no instance counts (effectively means K=10).
* For each row in rows 1-10, a random 90% of all columns are suppressed to 0.

Mean (standard deviation) of the residuals over the 100,000 runs are tabulated for both methods
| Tries |     SVD     |   Random    |
|:-----:|:-----------:|:-----------:|
|  1    | 0.20 (0.05) | 0.33 (0.08) |
|  10   | 0.13 (0.02) | 0.22 (0.04) |
|  100  | 0.09 (0.01) | 0.15 (0.02) |
|  1000 | 0.07 (0.01) | 0.11 (0.02) |

Here "SVD" is the proposed method and gives the best (lowest) residuals on average. The benchmark can be found in `Test_RandomSplit.py`.

## Cross Validation
Using the same setup as above, I calculated the residuals for F=3, 5 and 10 fold cross validation for both methods.

### F = 3
Mean (standard deviation) of the residuals over the 100,000 runs are tabulated for both methods
| Tries |     SVD     |   Random    |
|:-----:|:-----------:|:-----------:|
|  1    | 0.32 (0.08) | 0.37 (0.07) |
|  10   | 0.21 (0.03) | 0.27 (0.03) |
|  100  | 0.16 (0.02) | 0.22 (0.02) |
|  1000 | 0.13 (0.01) | 0.18 (0.02) |

### F = 5
Mean (standard deviation) of the residuals over the 100,000 runs are tabulated for both methods
| Tries |     SVD     |   Random    |
|:-----:|:-----------:|:-----------:|
|  1    | 0.32 (0.05) | 0.34 (0.05) |
|  10   | 0.24 (0.02) | 0.27 (0.02) |
|  100  | 0.21 (0.02) | 0.23 (0.02) |
|  1000 | 0.18 (0.01) | 0.21 (0.01) |

### F = 10
Mean (standard deviation) of the residuals over the 100,000 runs are tabulated for both methods
| Tries |     SVD     |   Random    |
|:-----:|:-----------:|:-----------:|
|  1    | 0.28 (0.04) | 0.29 (0.04) |
|  10   | 0.23 (0.02) | 0.24 (0.02) |
|  100  | 0.20 (0.01) | 0.21 (0.01) |
|  1000 | 0.18 (0.01) | 0.20 (0.01) |


# Remarks
This method helps in cases where the weight matrix W is sparse and N is not too *large*. If you have a dense weight matrix or N is *large*, simple random shuffling will probably give a good train/test split.

**NOTE** This remark is hinted by the following bound:

|ZDWx_train| <= sqrt(KN) sqrt(min(p, 1-p)) sqrt(sum_{n=1}^N var((DW)_n))

where (DW)_n is the n'th column of matrix DW and p is the training sample proportion. The bound is tighter when N is *small* and/or DW is sparse. Sparse DW results in columns with lower variance. The bound is tighter with a *small* number of criteria K although more criteria would likely result in lower column variance.

Here's another bound in terms of absolute difference between binary indicator vector **x**_train and approximate soft indicator vector a**x** + b**1**

|ZDWx_train| = |ZDW(a**x** + b**1** - x_train)| <= sqrt(K) sqrt(sum_{n=1}^N var((DW)_n)) |a**x** + b**1** - x_train| <= sqrt(KN) sqrt(sum\_{n=1}^N var((DW)_n))

The bound would be tight if your soft indicator vector is *nearly* a binary vector.

