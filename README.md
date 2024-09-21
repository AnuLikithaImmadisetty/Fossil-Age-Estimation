# Fossil Age Estimation" 🌍🦴
Welcome to the "Fossil Age Estimation" Repository! Here, we've explored an extensive range of around 30 machine learning models to predict fossil ages, utilizing a diverse set of geological and biological features! This project demonstrates the power of different machine learning techniques in estimating the age of fossils with remarkable precision, contributing to the understanding of Earth's ancient history.

Dive in to discover the models, analyses, and insights that emerged from this fascinating exploration!

## Table of Contents 
 - [Overview](#overview)
 - [Project Structure](#project-structure)
 - [Dataset and Description](#dataset-and-description)
 - [Features](#features)
 - [Tools and Technologies](#tools-and-technologies)
 - [Results](#results)
 - [Getting Started](#getting-started)
 - [Usage](#usage)
 - [Contributing](#contributing)
 - [License](#license)

## Overview 
In the intricate world of paleontology, estimating fossil ages with precision is crucial for understanding Earth's ancient history. The **Fossil Age Estimation** project leverages advanced machine learning techniques to predict the ages of fossils based on a diverse set of geological and biological features. This project aims to analyze these characteristics, identify key factors influencing fossil dating, and develop a predictive model capable of forecasting fossil ages with high accuracy. By combining data-driven insights with scientific inquiry, this project supports researchers in uncovering the timeline of life on Earth.

## Project Structure 
Here's an overview of the project's structure
```bash
│
├── Datasets/ : Contains datasets used for training, validation and testing.
├── Models/ : Code snippets for preprocessing, model training and evaluation.
├── Results/ : Contains output screenshots of performance metrics.
├── README.md : Explains an overview of the repository.
└── requirements.txt : Required the list of dependencies for this project.
```

## Dataset and Description 
This dataset consists of 24 columns and records the details of fossil specimens, with 9302 observations, representing a variety of geological and biological characteristics. The data captures vital attributes such as geographic locations, taxonomic classifications, dating techniques, and various stratigraphic and taphonomic details, all contributing to estimating fossil ages accurately.

The [Dataset](https://github.com/AnuLikithaImmadisetty/Fossil-Age-Estimation/blob/main/Datasets/fossil_data.csv) is available here! Below are the Descriptions of the columns available in the dataset:

- **ID:** A unique identifier assigned to each fossil specimen.
- **Latitude:** Latitude coordinates indicating the north-south position where the fossil was discovered.
- **Longitude:** Longitude coordinates indicating the east-west position of the fossil discovery.
- **Genus:** The genus classification of the fossilized organism.
- **Species:** The species classification of the fossilized organism.
- **Family:** The family to which the fossilized organism belongs.
- **Class:** The class of the fossilized organism in the biological taxonomy.
- **Infra-Class:** A subclassification within the broader class of the organism.
- **Order:** The order in the taxonomic hierarchy of the fossilized organism.
- **Status:** The current research status of the fossil specimen (e.g., whether it has been confirmed, pending, etc.).
- **Megafauna:** A binary indicator (Yes/No) showing whether the fossil belongs to a megafauna species.
- **Age ID:** A unique identifier for the fossil's estimated age.
- **Dated Remain:** The specific fossil remain that was dated (e.g., bone, tooth, etc.).
- **Dated Material:** The type of material that was used to determine the fossil’s age.
- **Age:** The estimated age of the fossil, usually in years.
- **Age Type:** The classification of the age (e.g., estimated, exact, etc.).
- **Dating Technique:** The technique used to date the fossil (e.g., radiocarbon dating, stratigraphy).
- **Overall Region:**  The broad geographic region where the fossil was discovered.
- **Administrative Division:** The administrative region or division (e.g., country, state) where the fossil was found.
- **Specific Region:** A more precise geographical description of where the fossil was located.
- **Cave/Site/Deposit:** The cave, site, or deposit in which the fossil was discovered.
- **Chamber/Provenance:** Specific chamber or location within the site from where the fossil was excavated.
- **Stratigraphy/Taphonomy:** Details about the fossil’s stratigraphy or taphonomic history.
- **Species abundances:** The number of fossils or abundance of the species in the excavation site.
- **Year:** The year the fossil was excavated or analyzed.
  
## Features 
The Fossil Age Estimation project incorporates the following components:
1. **🧪 Data Analysis**: Analyze fossil datasets to extract meaningful features for accurate age prediction.
2. **🧠 Machine Learning Models**: Implement, train, and optimize various around 30 machine learning models to estimate the age of fossils.
3. **📈 Evaluation Metrics**: Evaluate model performance using metrics like RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), R² and Explained Variance.

## Tools and Technologies
This project employs a diverse array of machine learning models to predict fossil ages. 

### Machine Learning Models:

🦴 **AdaBoostRegressor:** An ensemble technique that combines multiple weak regression models to form a strong predictive model, enhancing accuracy by focusing on misclassified data points.  
🌳 **BaggingRegressor:** A method that uses bootstrapping to create multiple subsets of the training data, training a separate model on each subset and averaging their predictions to reduce variance.  
📊 **BayesianRidgeRegressor:** A linear regression model that incorporates Bayesian inference, providing a probabilistic approach to regression with uncertainty quantification.  
🌱 **CatBoostRegressor:** A gradient boosting algorithm optimized for categorical features, capable of handling complex datasets while reducing overfitting.  
🌲 **DecisionTreeRegressor:** A model that splits the dataset into subsets based on feature values, creating a tree-like structure to make predictions. Its simplicity allows for easy interpretation.  
💡 **ElasticNetRegressor:** A linear regression model that combines L1 and L2 regularization, balancing between Ridge and Lasso regression to improve model performance on correlated features.  
🎲 **Extra Trees Regressor:** An ensemble method that builds multiple decision trees using random subsets of features, enhancing model robustness and accuracy.  
🔬 **Gradient Boosting Regression:** An ensemble technique that builds models sequentially, each one correcting errors of the previous model to improve predictive performance. It combines the predictions of several base estimators, typically decision trees, to produce a strong final model.  
🛠️ **HuberRegressor:** A regression model that is robust to outliers by combining the squared loss for small errors and absolute loss for larger errors, minimizing the impact of outliers on predictions.  
🌐 **IsotonicRegression:** A non-parametric regression technique that fits a free-form line to the data while ensuring that the predictions are monotonically increasing or decreasing.  
👥 **K-NearestNeighborsRegressor:** A non-parametric method that predicts the value of a sample based on the average value of its k-nearest neighbors in the feature space.  
🌀 **Kernel Ridge Regression:** A regression technique that combines Ridge Regression with the kernel trick, allowing it to model complex relationships by mapping input features into a high-dimensional feature space.  
🔄 **LassoLarsRegressor:** A linear regression model that applies L1 regularization using the Least Angle Regression algorithm, effectively performing variable selection and regularization.  
🧮 **LassoRegression:** A linear regression model that applies L1 regularization to minimize the sum of absolute errors, promoting sparsity in the feature set.  
⚡️ **LightGBM:** A gradient boosting framework that uses tree-based learning algorithms, optimized for speed and efficiency, particularly suited for large datasets.  
🗳 **LinearRegression:** A fundamental regression technique that models the relationship between input features and a continuous target variable using a linear equation.  
🛡️ **PassiveAggressiveRegression:** An online learning algorithm that updates the model incrementally with each new data point, combining passive and aggressive approaches for robustness.  
🎯 **Poisson Regression:** A model suited for count data, predicting the logarithm of expected counts, useful for data where the response variable represents counts or rates.  
🔢 **Polynomial Regression:** An extension of linear regression that models the relationship between the independent variable and the dependent variable as an nth-degree polynomial.  
🚧 **Quantile Regression:** A regression technique that estimates the conditional quantiles of the response variable, providing a more comprehensive view of the possible outcomes.  
🌳 **Random Forest Regression:** An ensemble method that constructs multiple decision trees and averages their predictions to enhance accuracy and control overfitting.  
🛡️ **Ridge Regression:** A linear regression model that incorporates L2 regularization to prevent overfitting by penalizing large coefficients.  
🔗 **Stacking Regression:** An ensemble technique that combines multiple regression models to improve predictive accuracy by learning how to best combine their predictions.   
🎲 **Stochastic Gradient Descent Regression (SGD):** A linear regression technique that uses stochastic gradient descent for optimization, allowing for efficient training on large datasets.  
🧠 **SVR (Support Vector Regression):** A regression technique that uses support vector machines to fit the best line within a specified margin, capable of handling non-linear relationships.   
🔍 **SVR with GridSearchCV:** A support vector regression model optimized using grid search to find the best hyperparameters for improved predictive performance.   
 🌈 **TransformedTargetRegressor:** A regression technique that applies a transformation to the target variable, improving model accuracy by adapting to data distributions.   
💡 **Tweedie Regression:** A generalized linear model that can predict outcomes that follow a Tweedie distribution, useful for modeling various types of data including counts and continuous outcomes.   
⚙️ **Voting Regression:** An ensemble method that combines predictions from multiple regression models, producing a final prediction based on majority voting or averaging.  
🚀 **XGBoost Regression:** An efficient and scalable boosting method that builds decision trees sequentially, correcting errors made by previous trees to enhance predictive performance. 

The [Models](https://github.com/AnuLikithaImmadisetty/Fossil-Age-Estimation/tree/main/Models) used here is available here!

## Results
Refer to the [Results](https://github.com/AnuLikithaImmadisetty/Fossil-Age-Estimation/tree/main/Results) here!

## Getting Started 
To get started with the Fossil Age Estimation project, follow these steps:
1. **Clone the Repository:** Clone the repository to your local machine using Git: (`git clone https://github.com/AnuLikithaImmadisetty/Fossil-Age-Estimation.git`)
2. **Navigate to the Project Directory:** Change to the project directory: (`cd Fossil-Age-Estimation`)
3. **Install Dependencies:** Install the required dependencies using pip: (`pip install -r requirements.txt`)
4. **Prepare the Data:** Download the dataset and place it in the `/Datasets` folder and ensure the data is in the expected format as described in the data documentation.
5. **Run the Analysis:** Open the Jupyter notebooks in Google Collab or Visual Studio Code located in the `/Models` folder, here you can explore various models and run the corresponding scripts to process the data, train the models, and make predictions.
6. **Evaluate the Models:** Review the evaluation metrics and results in the `/Results` folder. Metrics which will analyze the performance of the models.
  
## Usage 
To use the trained models for **Fossil Age Estimation**🌍:
1. Format and preprocess your text data to align with the training data specifications.
2. Utilize the provided notebooks in the `/Models` directory to load the trained model and generate predictions.
3. Examine the output to identify the age of the fossils.

## Contributing 
Contributions are welcome to this project. To contribute, please follow these steps:
1. Fork the Repository.
2. Create a New Branch (`git checkout -b feature/YourFeature`).
3. Make Your Changes.
4. Commit Your Changes (`git commit -m 'Add new feature'`).
5. Push to the Branch (`git push origin feature/YourFeature`).
6. Create a Pull Request.

## License 
This project is licensed under the MIT License. Refer to the LICENSE file included in the repository for details.
