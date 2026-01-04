# Credit Risk Data Prediction

## Repository Outline
- model_training.ipynb - File notebook berisi identitas pembuat, tujuan pembuatan model, visualisasi analisis dan code pembuatan model algoritma.
- model_inference.ipynb - File notebook berisi proses model inference model algoritma yang telah dibuat untuk memprediksi sebuah data baru


## Problem Background
Pihak Bank selalu dihadapi dengan risiko finansial saat peminjam mengalami default. Dari pihak Bank butuh suatu cara atau alat yang dapat mengurangi risiko ini. Dengan dibuatnya model ini, diharapkan model ini bisa memprediksi seorang peminjam risiko akan default atau tidak. Model ini bisa membantu mengurangi risiko finansial jika dari Bank bisa menggunakan model ini untuk menyeleksi seorang peminjam.

## Project Output
[Model Deployment](https://huggingface.co/spaces/ivan-carlos02/Credit-Risk-Prediction)

## Data
- Sumber Dataset - [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- 32581 baris
- 4011 missing values
- 12 kolom:
  - person_age: umur pengguna
  - person_income: pendapatan pengguna
  - person_home_ownership: jenis tempat tinggal apakah milik sendiri atau kontrak, dan lain-lain
  - person_emp_length: lama pengguna bekerja
  - loan_intent: tujuan pengguna meminjam
  - loan_grade: tingkat risiko pengguna
  - loan_amnt: jumlah yang dipinjam
  - loan_int_rate: bunga pinjaman
  - loan_percent_income: cicilan dari pendapatan
  - cb_person_default_on_file: apakah pengguna sebelumnya pernah default
  - cb_person_cred_hist_length: sejarah lama penggunaan credit si pengguna

## Method
- Model Training - Menggunakan 5 supervised classification model yaitu:
  - K-Nearest Neighbors (KNeighborsClassifier)
  - Support Vector Machine (SVC)
  - Decision Tree (DecisionTreeClassifier)
  - Random Forest (RandomForestClassifier)
  - Ada Boost (AdaBoostClassifier)
- Best Model - Cross Validation 
- Visualisasi - Histogram, Bar Chart, Pie Chart, BoxPlot, ConfusionMatrixDisplay
- Outlier Analisis - Z-Score dan Tukey's Rule, Winsorizer Capping method
- Missing Value - Imputation SimpleImputer
- Feature Scaling - StandardScaler
- Feature Encode - OneHotEncoder dan OrdinalEncoder
- Model Pipeline
- Model Evaluation - Recall dan Accuracy dari Classification Report, ROC-AUC score dan curve

## Stacks
- Pandas
- Scipy
- Matplotlib
- Seaborn
- Scikit-Learn
- Feature-Engine
- Streamlit
- Dill

## Reference
[Function Transformer Issue](https://onnx.ai/sklearn-onnx/auto_tutorial/plot_jfunction_transformer.html)