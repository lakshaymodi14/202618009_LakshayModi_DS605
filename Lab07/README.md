# DS605 Lab 7 — PCA & Eigenfaces

Face recognition using PCA, eigenfaces, and traditional machine-learning classifiers.

### Dataset
[AT&T Database of Faces (ORL) — Kaggle](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces)

Contains 400 grayscale face images of 40 subjects.

### Methodology
- Converted grayscale images into pixel feature vectors.
- Implemented PCA using **NumPy linear algebra** (`np.linalg.eigh` and `np.linalg.svd`), without Scikit-learn PCA.
- Retained 91 principal components, preserving 90.05% of the variance.
- Reconstructed faces and compared raw-pixel and PCA-based nearest-neighbor recognition.
- Compared PCA features using KNN, Logistic Regression, and SVM.

### Results

| Method | Accuracy |
|---|---:|
| Raw-pixel Nearest Neighbor | 96.67% |
| PCA Nearest Neighbor | 96.67% |
| KNN (k=1) | 96.67% |
| KNN (k=3) | 94.17% |
| Logistic Regression | 97.50% |
| **SVM (RBF)** | **98.33%** |

### Conclusion
PCA reduced the feature space from 10,304 pixels to 91 components while preserving 90.05% of the variance. SVM achieved the highest test accuracy of 98.33%.

### Run
Open `DS605_Lab7_PCA_Eigenfaces_Face_Recognition.ipynb` in Jupyter Notebook and run all cells after setting the dataset path.
