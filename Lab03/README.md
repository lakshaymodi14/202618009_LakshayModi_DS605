# DS605: Fundamentals of Machine Learning

## Lab Assignment 3 — Hotel Booking Cancellation Prediction

**Dataset:** [Kaggle Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

# Student ID- 202618009
# Name- Lakshay Modi

### Overview

This project uses the Hotel Booking Demand dataset to predict whether a hotel reservation will be cancelled (`is_canceled`). The focus is on comparing preprocessing choices and seeing how they affect two classification models.

### Methodology

* Removed `reservation_status` and `reservation_status_date` to avoid target leakage.
* Dropped `company` because of its very high missingness.
* Checked missing values and numerical outliers.
* Used a conservative 1st–99th percentile rule for selected extreme numerical values.
* Used an 80/20 stratified train-test split with `random_state=42`.
* Numerical missing values were handled using `KNNImputer(n_neighbors=5)`.
* Categorical missing values were filled using the most frequent value.
* Categorical variables were one-hot encoded.
* Compared:

  * **Pipeline A:** KNN Imputation + StandardScaler
  * **Pipeline B:** KNN Imputation + MinMaxScaler
* Models:

  * Logistic Regression (`max_iter=1000`)
  * Decision Tree (`random_state=42`)

This resulted in four model-pipeline combinations evaluated on the same test set.

### Results

| Model                                | Test Accuracy |  Precision |     Recall |   F1-Score |
| ------------------------------------ | ------------: | ---------: | ---------: | ---------: |
| Logistic Regression + StandardScaler |        81.16% |     79.60% |     65.24% |     71.71% |
| Logistic Regression + MinMaxScaler   |        80.78% |     79.30% |     64.25% |     70.99% |
| **Decision Tree + StandardScaler**   |    **85.91%** | **80.96%** | **80.38%** | **80.67%** |
| Decision Tree + MinMaxScaler         |        85.88% |     80.95% |     80.33% |     80.64% |

The **Decision Tree + StandardScaler** combination produced the best overall test performance.

The confusion matrix also shows a clear difference in cancellation detection. The Decision Tree correctly identified **6,860 canceled bookings** and had **1,674 false negatives**, compared with **5,568 true positives** and **2,966 false negatives** for Logistic Regression.

### Key Findings

* Decision Tree performed better than Logistic Regression across the main test metrics.
* StandardScaler gave Logistic Regression a small improvement over MinMaxScaler.
* Scaling had almost no effect on Decision Tree performance.
* Decision Tree reached **99.60% training accuracy** but **85.91% test accuracy**, giving a **13.69 percentage-point gap**, which indicates possible overfitting.
* Logistic Regression had much smaller train-test gaps, but its cancellation recall was lower.
* Since recall measures how many actual cancellations are detected, the Decision Tree's **80.38% recall** is particularly useful for identifying bookings that may be cancelled.

### Repository Contents

```text
DS605-Lab-3/
├── DS605_Lab_3_Hotel_Booking_Scikit.ipynb
├── hotel_bookings.csv
├── hotel_bookings_cleaned.csv
├── confusion_matrix_logistic_regression.png
├── confusion_matrix_decision_tree.png
└── README.md
```

### Conclusion

The Decision Tree gave the strongest predictive performance, especially for detecting canceled bookings. However, its large train-test gap shows that the higher performance comes with a risk of overfitting. Logistic Regression was more stable between training and testing but missed more actual cancellations.

Overall, the experiment shows that **model choice had a much larger effect than the choice between StandardScaler and MinMaxScaler for the Decision Tree**.

