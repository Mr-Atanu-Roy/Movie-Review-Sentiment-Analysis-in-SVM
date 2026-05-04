# **Comparative Study of SVM Kernels for Sentiment Classification**

In this project I have conducted a comparative and experiment-driven study of different SVM kernels for sentiment classification and analyzed why linear models outperform nonlinear kernels on high-dimensional TF-IDF representations.

- Find the detailed analysis report **[here](https://drive.google.com/file/d/1JftvsMSDil9cLCJARRXxftAOAzQ7Tvvl/view?usp=sharing)**.
- Code file can also accessed in kaggle **[here](https://www.kaggle.com/code/mratanuroy/movie-review-sentiment-analysis-in-svm)**.

## Overview

This project conducts an experiment-driven comparative study of three SVM kernel variants for binary sentiment classification on the **Stanford Large Movie Review Dataset (IMDB)**. The study focuses not only on building an accurate classifier but on empirically understanding model behavior — including why certain kernels fail entirely on sparse text data, and where even the best model hits a fundamental representational ceiling.

A full technical report covering methodology, results, discussion, and misclassification analysis is available **[here](https://drive.google.com/file/d/1JftvsMSDil9cLCJARRXxftAOAzQ7Tvvl/view?usp=sharing)**.

## Key Findings

| Model                     | Accuracy (%)   | F1 Score  | ROC-AUC   |
| ------------------------- | -------------- | --------- | --------- |
| **Linear SVM (C=0.1)**    | **88.72**      | **0.887** | **0.956** |
| RBF SVM                   | ~51            | 0.667     | 0.58      |
| Polynomial SVM            | ~50            | 0.667     | 0.56      |

- **Linear SVM** significantly outperforms nonlinear kernels across all metrics
- **RBF and Polynomial kernels** collapse to near-random guessing — predicting the positive class for virtually every sample
- **5-fold cross-validation** confirms stability: CV Accuracy = 89.06% ± 0.0036
- **Learning curve analysis** reveals a moderate overfitting gap of 0.047, shown to be irreducible through TF-IDF vocabulary tuning alone — pointing to the fundamental representational ceiling of bag-of-words features

## Repository Structure

```
assets/
 ├── docs/Documentation.pdf
 ├── images/
 └── outputs/
movie-review-sentiment-analysis.ipynb

```

Here:

- `assets/docs/Documentation.pdf` contains full techical report for this project
- `assets/images/` folder contains block diagram, plots, visuals
- `assets/outputs/` folder contains misclassification reports in csv for models
- `movie-review-sentiment-analysis.ipynb` is the main jupyter notebook file for this project

## Dataset Description

I have used the **[Stanford Large Movie Review Dataset (IMDB)](https://ai.stanford.edu/~amaas/data/sentiment/)** for this project.

**Dataset Description:**

- It contains 50,000 English movie reviews labeled as positive or negative
- The dataset is evenly balanced, with 25,000 positive and 25,000 negative reviews
- It is pre-split into 25,000 training and 25,000 testing samples.

## Methodology

The picture below is the block diagram of the aproach of the project

<div align="center" width="100%">
  <img src="./assets/images/SVM sentiment analysis_white.png" alt="Block Diagram" width="95%" />
</div>
 
The pipeline consists of five stages:
1. **Text Preprocessing** — Lowercasing, special character removal, stemming
2. **Feature Extraction** — TF-IDF vectorization with unigrams and bigrams, sublinear scaling
3. **Model Training** — 12 SVM models trained across 3 kernel types and multiple hyperparameter configurations
4. **Evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC
5. **Analysis** — Misclassification analysis, 5-fold cross-validation, learning curves, overfitting remediation experiments

## Results Summary

<div align="center" width="100%">
   <img src="./assets/images/all models metrices graph.png" alt="Block Diagram" width="95%" />
</div>
<br>
<div align="center" width="100%">
  <img src="./assets/images/all models metrices.png" alt="Block Diagram" width="95%" />
</div>
 
The complete results table, confusion matrices, ROC curves, cross-validation scores, and learning curve plots are documented in the **[technical report](https://drive.google.com/file/d/1JftvsMSDil9cLCJARRXxftAOAzQ7Tvvl/view?usp=sharing)**.
 
Misclassification reports (CSV) for the best model are available in [`assets/outputs/`](https://github.com/Mr-Atanu-Roy/Movie-Review-Sentiment-Analysis-in-SVM/tree/master/outputs).

## Limitations and Future Work

In this project I have focused on **classical machine learning** methods using TF-IDF features and Support Vector Machines. These approaches are effective, but they do not capture contextual or semantic information beyond word frequency.  
Future work could be on using **modern deep learning methods**, such as transformer-based models (e.g., BERT), which can model contextual relationships between words. Additionally,
experimenting with word embeddings or hybrid models may further improve sentiment classification performance.

## Author

- [@Mr-Atanu-Roy](https://github.com/Mr-Atanu-Roy)
