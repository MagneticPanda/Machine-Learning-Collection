# About
Machine Learning has been employed across various domains. In recent years, there has been a growth in computational predictions for various sports. One of these sports that is of particular interest is The National Basketball Association (NBA). Due to its immense global popularity, there have been numerous initiatives to gather data about various players and game, tracing back decades. This opulent source of data makes machine learning an appealing choice to detect outstanding players across and make predictions about the outcome of various games, amongst others. This report compares and contrasts various machine learning techniques in these two tasks. We utilise and evaluate various dimensionality reduction techniques (PCA and LLE), outlier detectors (SVM, IQR, Gaussian Mixtures), regressors (linear regression and polynomial regression), and a majority-voting ensemble of regressors. Through this evaluation we are able to conclude that the pairing of standard scaling LLE with the Gaussian Mixture anomaly was the best combination of techniques for detecting outstanding players. We also show that the ensemble technique is best for predicting game outcomes, with the strongest model in the ensemble being the polynomial regression model.

This repository contains the code, in the form of a Jupyer notebook, which was used to reach these conclusions.

## Methodology
![Task 1 tax](https://user-images.githubusercontent.com/71750671/200804108-2d7dcf13-5a4f-45e4-beb2-7855f31317cc.png)

![Task 2 Tax](https://user-images.githubusercontent.com/71750671/200804150-aeadaa74-19e1-4fb3-a67d-6975b1cc1b6d.png)

## How to Run
The *environment.yml* file contains a list of project dependencies which you can install using [Anaconda](https://www.anaconda.com/).
To do this, simply execute the following command on your system:

```shell
conda env create --file environment.yaml
```
