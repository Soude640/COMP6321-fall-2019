# Dataset: *Bike Sharing* (Regression 6)

## 1. Total running time: 

about __ minutes


## 2. Results 

### 2.1 Support Vector Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| C                       | 1.0           | 10.453343880632431  |
| gamma                   | 'auto'        | 'auto' |
| kernel                  | 'rbf'         | 'linear'         |
|                         |               |               |
| **mean_sqaured_error**  | 3762.34570       | 0.00285       |
| **R^2_score**           | 0.88610     | 1.00000       |

### 2.2 Decision Tree Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| max_depth               | None          | 5             |
| min_samples_leaf        | 1             | 7             |
|                         |               |               |
| **mean_sqaured_error**  | 36.10460       | 34.32476       |
| **R^2_score**           | 0.99891     | 0.99896       |


### 2.3 Random Forest Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| n_estimators            | 10            | 1224          |
| max_depth               | None          | 23            |
|                         |               |               |
| **mean_sqaured_error**  | 33.70461      | 11.68101      |
| **R^2_score**           | 0.99891     | 0.99965       |


### 2.4 Ada Boost Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| n_estimators            | 50            | 10400            |
| learning_rate           | 1.0           | 0.010105000207208205     |
|                         |               |               |
| **mean_sqaured_error**  | 614.17270       | 614.17270      |
| **R^2_score**           | 0.98141     | 0.98141       |

*(Comment: Training time is extreme long!(about 5min per run))*


### 2.5 Gaussian Process Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| alpha                   | 1e-10         | 0.5440077     |
| kernel                  | 1**2 * RBF(length_scale=1)  | 1**2 * RBF(length_scale=0.5)              |
|                         |               |               |
| **mean_sqaured_error**  | 1981.61153       | 0.42887       |
| **R^2_score**           | 0.94001     | 0.98141       |


### 2.6 Linear Least Squares

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| alpha                   | 1.0           | 13.528105     |
| max_iter                | None          | 144           |
| solver                  | 'auto'        | 'saga'        |
|                         |               |               |
| **mean_sqaured_error**  | 0.44277       | 0.44089       |
| **R^2_score**           | 0.94001     | 0.98141       |


### 2.7 Neural Network Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| hidden_layer_sizes      | 100           | 814           |
| max_iter                | 200           | 1270          |
|                         |               |               |
| **mean_sqaured_error**  | 0.73434       | 0.45231       |
| **R^2_score**           | 0.94001     | 0.98141       |

