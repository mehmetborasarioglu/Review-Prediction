
# Movie Review Sentiment Classification

This repository contains code and documentation for a classical machine learning pipeline developed to predict star ratings of movie reviews. The project was conducted under strict constraints: no use of neural networks or boosting libraries. The main goal was to explore the capabilities of classical models such as Logistic Regression and Linear SVM on high-dimensional sparse data.

## Dataset Overview

The dataset comprises user-written movie reviews, with accompanying metadata including user ID, product ID, and star ratings. After vectorization using TF-IDF and other feature engineering techniques, the final feature matrices were extremely sparse and high-dimensional.

Train set shape: (1,188,272, 1,090,799)  
Validation set shape: (297,069, 1,090,799)  
Test set shape: (212,192, 1,090,799)  

All matrices were represented in sparse format to optimize memory usage.

I explored the data using many different techniques, I looked at the correlation between the sentiment values of text and summary, the text and summary length for each star score. I also created wordclouds each star reviews text and summary files to see if there were major difference between the words that are being used. Also observed that the data was highly skewed towards 5 star reviews so i kept that in mind aswell.

## Feature Engineering Attempts for User and Product IDs

### 1. Correlation Matrices

We initially attempted to calculate correlation matrices between users and products based on their rating patterns. These matrices included cosine similarity and Pearson correlation. However, due to the massive number of users and products, the resulting matrices were extremely large, leading to high memory usage and computational inefficiency. This approach was eventually abandoned.

### 2. One-Hot Encoding

We then tried one-hot encoding for user and product IDs. While this approach added minimal computational complexity during preprocessing, it drastically increased the dimensionality of the already sparse feature matrix. This caused a significant slowdown in training time and did not yield better performance. Hence, this approach was reverted.

### 3. TruncatedSVD Embeddings

Another approach involved building a user-product interaction matrix and applying TruncatedSVD to reduce dimensionality. While theoretically sound, this method suffered from cold-start problems and overfitting. The embeddings were not robust enough to generalize well to the test set. Also turned out to be computationally difficult. As a result, this method was discarded.

### 4. ALS Embeddings from the Implicit Library

We used the Implicit library to create Alternating Least Squares (ALS) embeddings. These low-dimensional vectors were intended to represent latent preferences of users and characteristics of products. Despite the theoretical advantages, this method also failed to improve model performance due to the same cold-start issues and the limited number of interactions for most users and items.

### 5. Target Encoding

We finally adopted target encoding (mean encoding) for user and product IDs. Each user was replaced with the bias score I have calculated for them which represented how harsh or soft they were compared the the average user and then product ID was replaced by the mean star rating associated with them in the training set. This approach proved simple yet highly effective, capturing user bias and product popularity without inflating the feature space. It had minimal overfitting risk and worked well with regularized linear models.

### 6. Target Encoding
Then I oversampled the undersrepresented data to be able to generelize better. 

## Modeling Attempts

### LinearSVC

We first attempted to train a LinearSVC model on the sparse feature matrix. However, this model uses the liblinear solver, which does not handle sparse input efficiently for very high-dimensional data. Internally, it densifies the matrix, resulting in extreme memory usage and very long training times. In most attempts, the model failed to converge or caused the system to hang. We reverted this model.

### LogisticRegression

We tried using LogisticRegression with the 'saga' solver, which supports sparse input and is suitable for large datasets. This model performed better than LinearSVC in terms of scalability and convergence. However, training was still slow due to the size of the data. While viable, we later opted for a more efficient alternative.

### SGDClassifier

We ultimately adopted SGDClassifier with both 'hinge' loss (for linear SVM) and 'log_loss' (for logistic regression). This model is optimized for large-scale learning and handles sparse matrices efficiently. It converged quickly and scaled well with our dataset. It also allowed for easy tuning via regularization and early stopping. This was the final model we used for training and evaluation.

## Modeling Tuning

While I was still deciding on which model to use I used gridsearch to tune the model parameters which was pretty straightforward.

## Modeling Evaluation
In this step, I applied 5-fold cross-validation with a linear SVM on the oversampled training data and evaluated the predictions using accuracy, confusion matrix, and a classification report to assess model performance across balanced folds. However this took a very long time to run in general.

## Struggles and Issues
Faced challenges with class imbalance — lower-rated reviews were underrepresented, impacting model accuracy.
Dimensionality of TF-IDF was very high — needed to tune max features.
Unsure how well the handcrafted features contributed — limited time to test feature importance in detail.
Considered using ensemble models or boosting, but restricted by competition rules.
Hyperparameter tuning was time-consuming; used a coarse grid due to runtime limits.
Tried a lot of ways to embed the users and products via collabrative filtering which took a lot of effort but at the end they didn't affect the accuracy.


## Dependencies

To run this project, the following Python libraries are required:

- scikit-learn
- pandas
- numpy
- scipy
- category_encoders
- implicit (only required if testing ALS embeddings)

Install dependencies using pip:

```
pip install -r requirements.txt
```

## How to Run

Run every cell after data extraction in starter_code.ipynb

The scripts are designed to load the sparse feature matrices and labels from preprocessed files and train the final model efficiently using SGDClassifier.



## Final Notes

After multiple iterations and tests with various feature engineering and modeling techniques, the most effective combination was TF-IDF for text data, target encoding for user/product IDs, and training with LinearSVC. This pipeline offered the best trade-off between training speed, memory efficiency, and prediction accuracy on the validation set.
