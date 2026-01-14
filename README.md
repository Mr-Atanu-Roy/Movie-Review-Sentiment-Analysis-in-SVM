# **Comparative Study of SVM Kernels for Sentiment Classification**

In this project I have conducted a comparative and experiment-driven study of different SVM kernels for sentiment classification and analyzed why linear models outperform nonlinear kernels on high-dimensional TF-IDF representations.

-   Find the detailed analysis report **[here](https://github.com/Mr-Atanu-Roy/Movie-Review-Sentiment-Analysis-in-SVM/blob/master/assets/docs/Documentation.pdf)**.
-   Code file can also accessed in kaggle **[here](https://www.kaggle.com/code/mratanuroy/movie-review-sentiment-analysis-in-svm)**.

## AIM of this project?

**The goal is to go beyond accuracy and understand model behavior.**

I have done a comparative study on 3 SVM variants on the IMDB sentiment dataset:

-   Linear SVM
-   RBF (Radial Basis Function) Kernel SVM
-   Polynomial Kernel SVM

Each model was evaluated using:

-   Accuracy
-   Precision, Recall, F1
-   Confusion Matrix
-   ROC-AUC
-   Misclassification analysis

## Folder Structure

```
assets/
 ├── docs/Documentation.pdf
 ├── images/
 └── outputs/
movie-review-sentiment-analysis.ipynb

```

Here:

-   `assets/docs/Documentation.pdf` contains full techical report for this project
-   `assets/images/` folder contains block diagram, plots, visuals
-   `assets/outputs/` folder contains misclassification reports in csv for models
-   `movie-review-sentiment-analysis.ipynb` is the main jupyter notebook file for this project

## Dataset Description

I have used the **[Stanford Large Movie Review Dataset (IMDB)](https://ai.stanford.edu/~amaas/data/sentiment/)** for this project.

**Dataset Description:**

-   It contains 50,000 English movie reviews labeled as positive or negative
-   The dataset is evenly balanced, with 25,000 positive and 25,000 negative reviews
-   It is pre-split into 25,000 training and 25,000 testing samples.

## Methodology

The picture below is the block diagram of the aproach of the project

<div align="center" width="100%">
  <img src="./assets/images/SVM sentiment analysis_white.png" alt="Block Diagram" width="95%" />
</div>

1. **Text Preprocessing**
   Raw reviews texts are preprocessed to reduce noise and
   standardize text. This included lowercasing, removal of unnecessary characters, perform
   stemming.

2. **Feature Extraction**

-   **Purpose:** Machine learning models operate on numerical data, so textual reviews are transformed into numerical feature vectors
-   **Method Used**: TF-IDF vectorization is used for this purpose. Term Frequency–Inverse Document Frequency (TF-IDF) converts text data to numerical data by assigning higher weights to words that are frequent within a review but rare across the corpus, making it effective for capturing sentiment-bearing terms. Both unigrams and bigrams were used to capture local context such as negations. Sublinear term frequency scaling was applied to reduce the influence of very frequent words in long reviews.

3. **Model Selection**
   Support Vector Machines (SVMs) are chosen due to their strong performance in high dimensional spaces and their effectiveness in text classification tasks. Three variants are evaluated: **Linear SVM**, **SVM with Radial Basis Function (RBF) kernel** and **SVM with Polynomial kernel**. These kernels differ in how they model decision boundaries.

4. **Hyperparameter Tuning**

    The following hyperparameters were tuned:

-   **C**: Regularization parameter.
-   **Gamma**: Kernel coefficient (RBF & Poly kernels).
-   **Degree**: Highest power of the polynomial used to form decision boundary (for polynomial kernel only).

The values they were tuned to are in detailed mentioned in the results section below and on **[the detailed report](https://github.com/Mr-Atanu-Roy/Movie-Review-Sentiment-Analysis-in-SVM/blob/master/assets/docs/Documentation.pdf)** along with their observed results.

1. **Training**
   Based on the above defined configurations a total of 12 SVMs were trained. Picture below shows the various matrices obtained after training.

<div align="center" width="100%">
  <img src="./assets/images/all models metrices.png" alt="Block Diagram" width="95%" />
</div>

<div align="center" width="100%">
   <img src="./assets/images/all models metrices graph.png" alt="Block Diagram" width="95%" />
</div>

## Results

1. **Linear SVM**

-   4 linear SVMs we trained with 4 different values of hyperparatemer `C`: `C=0.01, 0.1, 1, 10`.
-   **Linear SVM achieved the best overall performance among all models**. The optimal value of the regularization parameter was `C = 0.1`, which resulted in the highest F1-score (0.887) and ROC–AUC (0.956). At this value of C, Precision and Recall were also highly and balanced.
-   Performance improved as `C` increased from **0.01 to 0.1** and then degraded after that for larger values, indicating a tradeoff between bias and variance.

2. **RBF Kernel SVM**

-   RBF SVM models were trained on a reduced subset of the training data due to high computational cost.
-   4 Linear SVM models with RBF kernels were trained with different values of `C` and `gamma`: `C=0.1 & gamma=0.001`, `C=1 & gamma=0.01`, `C=0.1 & gamma=0.01`, `C=1 & gamma=0.001`.
-   All models gave similar results for all tested values of `C` and `gamma`. The model achieved accuracy close to random guessing **(~51%)**, with very high recall **(0.99)** but very low precision **(0.50)**. ROC–AUC values remained around **0.69**, indicating weak class separation.
-   Very high recall and low precision suggests that the model is predicting almost everything as positive label, indicating unstable decision behavior.

3. **Polynomial Kernel SVM**

-   Polynomial SVM models were trained on a reduced subset of the training data due to high computational cost.
-   4 Linear SVM models with RBF kernels were trained with different values of C and degree: `C=0.1 & degree=2`, `C=1 & degree=2`, `C=0.1 & degree=3`, `C=1 & degree=3`.
-   All the Polynomial SVM models tested across all values of `C` and `degree` showed near-random performance across all tested configurations. Accuracy remained close to **50%**, and ROC AUC values were approximately **0.56**.
-   High recall combined with low precision suggests that the model is predicting almost everything as positive label, indicating unstable decision behavior.

To better understand the limitations of the best-performing Linear SVM model (C = 0.1), I have conducted an error analysis using the misclassified samples from the test set. A separate report was generated containing the raw and processed reviews along with basic text statistics such as number of characters, words, and sentences.
The missclasification reports can be found **[here](https://github.com/Mr-Atanu-Roy/Movie-Review-Sentiment-Analysis-in-SVM/tree/master/outputs)**.

## Conclusion

1. **Why Linear Kernel Performed Best?**
   The Linear SVM achieved the best performance among all evaluated models. This is because TF-IDF representations of text data are high-dimensional and sparse, and in these cases linear classifiers are known to perform well. In such feature spaces, positive and negative sentiment reviews often become linearly separable. Linear SVM effectively uses this property by learning a stable decision boundary that generalizes well to unseen data while remaining computationally efficient.

2. **Why RBF and Polynomial Kernels Failed?**
   SVMs with RBF and Polynomial kernels performed significantly poor than Linear SVM in this project. These kernels rely on distance-based similarity measures or complex feature interactions, which are not well suited for sparse TF-IDF vectors that is created for reviews here. In high-dimensional text spaces, distance metrics become unreliable and noise is easily amplified. As a result, both RBF and Polynomial SVMs struggled to create a meaningful decision boundary and often collapsed into predicting a single class (positive in my case). Due to this the model is just good as random guesses.

3. **Misclassification Analysis**
   After analyzing the misclassified reviews, I found the following things:

-   Many misclassified reviews are: long and contains both positive and negative expressions.
-   Phrases involving: “not”, “but”, “however” are misclassified as TF-IDF represents text as independent weighted tokens, it cannot fully capture the scope of negation or contrast, leading to incorrect sentiment interpretation.
-   Several misclassified reviews are mildly positive or mildly negative.
-   Some reviews contain rare words. Since I have set the min df=5, so rare phrases may not survive TF-IDF filtering.
-   Some reviews are extremely short.
-   Misclassifications occurred across wide range of reviews across all length and types. This indicates that misclassification errors are not strongly correlated with review size but rather with linguistic complexity and contextual structure.

## Limitations and Future Work

In this project I have focused on **classical machine learning** methods using TF-IDF features and Support Vector Machines. These approaches are effective, but they do not capture contextual or semantic information beyond word frequency.  
Future work could be on using **modern deep learning methods**, such as transformer-based models (e.g., BERT), which can model contextual relationships between words. Additionally,
experimenting with word embeddings or hybrid models may further improve sentiment classification performance.

## Author

-   [@Mr-Atanu-Roy](https://github.com/Mr-Atanu-Roy)
