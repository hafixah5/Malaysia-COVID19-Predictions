# Malaysia-COVID19-Predictions
 
## Table of Contents

* [Project Description](#project-description)
* [Project Files](#project-files)
* [Project Usage](#project-usage)
* [Credit](#credit)

##  :scroll:  Project Description

This project aims to create a deep learning model to predict the umber of new cases in Malaysia.
For this objective, a few variables were taken from patients.
The data collected can be retrieved in the dataset folder in the repo.

This model achieved a low error in predicting the number of new cases.
There are a few missing data which was then interpolated to fill them. Polynomial interpolation was used as it is believed to be more accurate.
MinMax Scaler was used to scale the data.

Example of **missing** data:

![missingnewcases_df_test.png](https://github.com/hafixah5/Malaysia-COVID19-Predictions/blob/main/images/missingnewcases_df_test.png)

Example of **filled** data after interpolation:

![fillednewcases_df_test.png](https://github.com/hafixah5/Malaysia-COVID19-Predictions/blob/main/images/fillednewcases_df_test.png)

The model developed is as below:

![model architecture.png](https://github.com/hafixah5/Malaysia-COVID19-Predictions/blob/main/images/model%20architecture.png)

For time-series data, errors are usually taken to determine the model accuracy, by finding the difference in prediction to the actual value.
Forecast errors are different to residuals as residual is calcualted on training data, while forecast errors are calculated on test data ([Source link](https://otexts.com/fpp2/accuracy.html))

From this project, the model is able to produce low errors. The errors measured are Mean Absolute Error (MAE), Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE).

Model error metrics:

![model error metrics.png](https://github.com/hafixah5/Malaysia-COVID19-Predictions/blob/main/images/model%20error%20metrics.png)

##  :card_index_dividers:  Project Files

:point_right: dataset folder
- cases_malaysia_train.csv
- cases_malaysia_test.csv

:point_right: COVID19_prediction.py file

:point_right: modules_covid.py file

:point_right: saved_files folder that contains encoded features

:point_right: images folder which contains the following images:
- model architecture
- plots of missing and filled data points
- plots of actual vs predicted data (scaled and unscaled data) 
- screenshots from Tensorboard (epoch loss, mae and mse)
- screenshots of model parameter and value of error metrics


##  :rocket:  Project Usage
1) This project is done using Python 3.8 on Spyder.
This project used the following modules:

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

2) The sample datasets, model and module have already been included in the repository.

3) You may download all the necessary files (dataset & python files) to run the project on your device.

## :technologist:  Credit

This dataset is taken from: [MOH Malaysia | GitHub](https://github.com/MoH-Malaysia/covid19-public)
