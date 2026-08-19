# Academic Efficiency: predicting focus and exam performance from student habits

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-8A2BE2?style=for-the-badge&logo=python&logoColor=white)

> 📖 Quick note in Portuguese: You can also read this README in Portuguese. To do so, just access [here](README.pt.md).

## About the project

This project analyzes a synthetic dataset of 5,000 student records (`Dados_Projeto/student_records_missing.csv`, with a complete reference version in `student_records_full.csv`) to study **academic efficiency** — whether exam performance depends only on study hours or on a broader combination of factors such as focus, rest, digital habits and overall well-being. It was developed for the **Machine Learning Fundamentals** (Fundamentos de Aprendizagem Automática) course, and defines two complementary prediction tasks: a **regression** problem estimating `focus_index` (a proxy for concentration while studying) and a **classification** problem predicting `high_exam_score`, a binary target built from the top quartile of `exam_score`.

The project was developed by Group 1: Guilherme Soares, Duarte Soares and Vitória Correia.

### Features
- Data loading, target construction (`high_exam_score` derived from the `exam_score` upper quartile) and exploratory inspection of numerical/categorical variables and missing values
- A reproducible preprocessing `Pipeline`/`ColumnTransformer`: median imputation + `RobustScaler` for numeric features, most-frequent imputation + one-hot encoding for categorical features, fit only on training data to avoid data leakage
- Baseline models — Linear Regression and Decision Tree for the regression task, Logistic Regression and Decision Tree for the classification task — evaluated with train/test split and cross-validation (`KFold`/`StratifiedKFold`)
- Model evaluation with RMSE/MAE for regression and F1-score/ROC-AUC/accuracy for classification, plus confusion matrices, coefficient plots and decision tree visualizations
- Error analysis inspecting the largest regression residuals and the false positives/negatives of the classifier
- Feature engineering and selection: correlation filtering, variance threshold, and embedded importance-based selection (AdaBoost for classification, Gradient Boosting for regression) with cross-validated search for the optimal number of features
- More complex models (MLPClassifier, Random Forest, Gradient Boosting) with systematic hyperparameter tuning via `RandomizedSearchCV`/`GridSearchCV`
- Unsupervised learning: K-Means clustering (with elbow/silhouette analysis) to uncover latent student "personas" based on behavioral and well-being attributes, without using the target variables
- Model interpretability with SHAP, comparing local explanations between Random Forest and Logistic Regression on true/false positive/negative cases

### Tech stack
- Python (Jupyter Notebook)
- Pandas / NumPy
- scikit-learn (pipelines, linear models, trees, ensembles, MLP, clustering, model selection, feature selection, metrics)
- Matplotlib / Seaborn
- SciPy (`loguniform` for randomized search distributions)
- SHAP
