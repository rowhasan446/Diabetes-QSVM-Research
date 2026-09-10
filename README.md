# 🩺 Diabetes Risk Analysis System Using QSVM

A quantum machine learning-based diabetes risk analysis system using Quantum Support Vector Machine (QSVM), patient-specific SHAP explanations, and Double Machine Learning (DML) research analysis.

## 🚀 Project Overview

This project explores quantum machine learning for diabetes risk prediction using two independent assessment pathways:

- 🩺 Clinical Data
- 🧠 Symptoms

Each pathway uses its own independently trained QSVM model.

## 🩺 Clinical QSVM

The clinical model uses five selected features:

1. Family history of diabetes
2. Hypertension
3. Blood glucose
4. BMI
5. Age

The model uses:

- StandardScaler
- ZFeatureMap
- FidelityQuantumKernel
- QSVC

## 🧠 Symptom QSVM

The symptom model uses seven selected features:

1. Polyuria
2. Polydipsia
3. Sudden weight loss
4. Partial paresis
5. Polyphagia
6. Gender
7. Irritability

The model uses:

- StandardScaler
- ZFeatureMap
- FidelityQuantumKernel
- QSVC

## 🔬 Explainable AI

Patient-specific SHAP explanations are provided using surrogate Random Forest models that approximate the QSVM decision function.

Therefore, SHAP explanations describe the surrogate approximation of the QSVM decision function and should not be interpreted as causal effects.

## 📈 Double Machine Learning

The symptom branch includes population-level Double Machine Learning (DML) analysis.

DML is presented as research context and is not used to modify the QSVM prediction.

DML estimates are observational and should not be interpreted as proof that a particular symptom causes diabetes in an individual.

## 🩺 Personalized Feedback

The application follows this pipeline:

QSVM prediction → Patient-specific SHAP → Controlled guidance → Personalized feedback

The feedback system identifies important model-influencing features and provides controlled, non-diagnostic guidance.

It does not diagnose disease or prescribe treatment.

## 🏗️ System Architecture

```text
                 DIABETES RISK ANALYSIS SYSTEM
                           |
              +------------+------------+
              |                         |
        Clinical Data              Symptoms
              |                         |
        Clinical QSVM             Symptom QSVM
              |                         |
              +------------+------------+
                           |
                    Risk Prediction
                           |
                 +---------+---------+
                 |         |         |
                QSVM      SHAP      DML
                 |         |         |
                 |    Individual   Population
                 |    reasoning     context
                 +---------+---------+
                           |
                 Personalized Feedback
```

## 📁 Repository Structure

```text
Diabetes-QSVM-Research/
├── app/
│   ├── app.py
│   ├── requirements.txt
│   └── models/
│       ├── clinical/
│       └── symptoms/
├── notebooks/
├── README.md
└── .gitignore
```

## ⚙️ Technologies

- Python
- Streamlit
- Qiskit
- Qiskit Machine Learning
- QSVC
- Fidelity Quantum Kernel
- ZFeatureMap
- SHAP
- Random Forest
- Double Machine Learning
- Pandas
- NumPy
- Scikit-learn

## 🔬 Scientific Notes

The two QSVM branches are independent models trained on different feature spaces and datasets.

The application does not retrain the models during prediction.

Additional information fields shown in the application are contextual only and do not affect the QSVM prediction unless they are among the model's required features.

### Important distinction

- QSVM: predicts diabetes risk according to the trained model.
- SHAP: explains the model decision through a surrogate approximation.
- DML: provides population-level observational research estimates.
- Personalized feedback: translates model findings into controlled guidance.

## ⚠️ Medical Disclaimer

This application is a research and educational machine-learning system.

Its predictions are not a medical diagnosis and should not replace professional medical advice, laboratory testing, or clinical evaluation.

If you have persistent, severe, new, or worsening symptoms, seek appropriate medical care.

## 👨‍💻 Project

Developed as a research/capstone project exploring quantum machine learning and explainable AI for diabetes risk analysis.