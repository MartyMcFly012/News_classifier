## Overview

This project is aimed at classifying news articles as either fake or real using machine learning and Python. The dataset contains a collection of labeled news articles, and I use the scikit-learn library to build a logistic regression classifier. Additionally, I visualize the model's performance using a confusion matrix heatmap (pictured below).

![Confusion matrix visualization](confusion_matrix_visualization.png)

## Dataset

The dataset used in this project is available on Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

## Required libraries

Bash: ```pip install numpy pandas scikit-learn matplotlib seaborn```

## Usage

Run the news_classification.py script to train the logistic regression classifier and evaluate its performance.

Bash: ```python news_classification.py```

## Files and directory structure

news_classification.py: The main Python script for data loading, preprocessing, feature extraction, model training, evaluation, and visualization.
fake_news.csv: CSV file containing fake news articles data.
real_news.csv: CSV file containing real news articles data.
README.md: This file, providing project information and instructions.
LICENSE: The project's license (MIT).
