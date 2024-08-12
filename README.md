# Movie Sentiment Analysis using ML & Spark

This repository contains a comprehensive project focused on building, tuning, and evaluating various machine learning models using **PySpark** and **scikit-learn** for movie sentiment analysis. The models included in this project are:

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest**
- **Word2Vec with Random Forest**
- **XGBoost**
- **Deep Learning Model**

## Project Overview

The goal of this project is to classify movie reviews as positive or negative by comparing different machine learning algorithms across various performance metrics such as **F1 score**, **precision**, **recall**, and **accuracy**. We utilize **feature extraction**, **model training**, **parameter tuning**, **cross-validation**, and rigorous evaluation to determine the best-performing model for our dataset.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Parameter Tuning](#parameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started, clone the repository and install the required dependencies:

git clone https://github.com/yourusername/movie-sentiment-analysis.git
cd movie-sentiment-analysis
pip install -r requirements.txt

## Project Structure

```bash
├── data
│   ├── raw_data.csv
│   └── processed_data.csv
├── notebooks
│   ├── feature_extraction.ipynb
│   ├── model_training.ipynb
│   └── model_evaluation.ipynb
├── models
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   ├── random_forest.pkl
│   ├── word2vec_random_forest.pkl
│   ├── xgboost.pkl
│   └── deep_learning_model.h5
├── scripts
│   ├── feature_extraction.py
│   ├── train_models.py
│   └── evaluate_models.py
└── README.md
```

## Feature Extraction
The feature extraction step involves processing raw data to create a set of features that can be used for model training. Techniques such as Word2Vec are used to generate word embeddings for text data, and other preprocessing steps like scaling and encoding are applied to structured data.

## Model Training
Each machine learning model is trained on the processed dataset using PySpark and scikit-learn. The models are optimized using techniques like Grid Search and Random Search to find the best hyperparameters.

## Parameter Tuning
We perform parameter tuning using cross-validation to ensure that the models are not only accurate but also generalize well to unseen data. This section of the project involves detailed exploration of the hyperparameter space for each model.

## Model Evaluation
After training, the models are evaluated on a test set using metrics like F1 score, precision, recall, and accuracy. Comparative analysis is done to highlight the strengths and weaknesses of each model.

## Results
The results of the model evaluations are summarized in this section. We present the performance metrics for each model and discuss the trade-offs between different approaches.

## Future Work
- Explore additional models such as Gradient Boosting Machines or Neural Networks.
- Implement more advanced feature engineering techniques.
- Optimize models further by implementing Bayesian Optimization for hyperparameter tuning.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with your suggestions.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
