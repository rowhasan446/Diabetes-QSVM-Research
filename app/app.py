
import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
import shap


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetes Risk Analysis",
    page_icon="🩺",
    layout="wide"
)

# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLINICAL_DIR = os.path.join(
    BASE_DIR, "models", "clinical"
)

SYMPTOM_DIR = os.path.join(
    BASE_DIR, "models", "symptoms"
)

# ============================================================
# HELPER
# ============================================================

def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)






# ============================================================
# CLINICAL PERSONALIZED GUIDANCE
# ============================================================

clinical_guidance = {

    "glucose": {
        "name": "Blood glucose",
        "type": "clinical_measurement",
        "priority": "High",
        "guidance": (
            "Blood glucose is an important component of diabetes-risk "
            "assessment. Abnormal glucose measurements should be "
            "appropriately monitored and discussed with a healthcare "
            "professional."
        )
    },

    "bmi": {
        "name": "Body mass index (BMI)",
        "type": "modifiable",
        "priority": "Moderate",
        "guidance": (
            "Maintaining a healthy body weight can support overall "
            "metabolic health. Appropriate nutrition and regular "
            "physical activity may be helpful when suitable for "
            "the individual."
        )
    },

    "hypertensive": {
        "name": "Hypertension",
        "type": "modifiable",
        "priority": "High",
        "guidance": (
            "Blood pressure is an important health factor. If you "
            "have high blood pressure, regular monitoring and "
            "appropriate management with guidance from a healthcare "
            "professional are recommended."
        )
    },

    "age": {
        "name": "Age",
        "type": "non_modifiable",
        "priority": "Context",
        "guidance": (
            "Age is a non-modifiable characteristic. It should not "
            "be treated as something that needs to be changed. "
            "Instead, it provides context for overall risk assessment."
        )
    },

    "family_diabetes": {
        "name": "Family history of diabetes",
        "type": "non_modifiable",
        "priority": "Context",
        "guidance": (
            "Family history cannot be changed. However, awareness "
            "of family history can support appropriate diabetes-risk "
            "screening and health monitoring."
        )
    }
}


# ============================================================
# CLINICAL FEEDBACK GENERATOR
# ============================================================

def generate_clinical_feedback(
    prediction,
    clinical_explanation
):
    """
    Generate controlled, personalized informational feedback.

    The feedback uses:
        - QSVM prediction
        - patient-specific SHAP values
        - predefined clinical guidance

    It does NOT diagnose, prescribe, or modify the QSVM result.
    """

    feedback_sections = []

    # --------------------------------------------------------
    # Risk result
    # --------------------------------------------------------

    if prediction == 1:

        feedback_sections.append(
            "🔴 **Higher Predicted Diabetes Risk**\n\n"
            "The clinical QSVM classified the current clinical "
            "profile as higher predicted diabetes risk."
        )

    else:

        feedback_sections.append(
            "🟢 **Lower Predicted Diabetes Risk**\n\n"
            "The clinical QSVM classified the current clinical "
            "profile as lower predicted diabetes risk."
        )


    # --------------------------------------------------------
    # Positive SHAP contributors
    # --------------------------------------------------------

    positive_factors = clinical_explanation[
        clinical_explanation["SHAP"] > 0
    ].copy()

    if len(positive_factors) > 0:

        positive_factors = positive_factors.sort_values(
            "SHAP",
            ascending=False
        )

        why_text = (
            "### 🧠 Why did the model make this prediction?\n\n"
        )

        for _, row in positive_factors.iterrows():

            feature = row["Feature"]
            shap_value = row["SHAP"]

            feature_name = clinical_guidance.get(
                feature,
                {}
            ).get(
                "name",
                feature
            )

            why_text += (
                f"🔴 **{feature_name}** — positive model "
                f"influence (`{shap_value:.4f}`)\n\n"
            )

        feedback_sections.append(why_text)

    else:

        feedback_sections.append(
            "### 🧠 Why did the model make this prediction?\n\n"
            "No positive SHAP contributors were identified "
            "for this patient profile."
        )


    # --------------------------------------------------------
    # What deserves attention?
    # --------------------------------------------------------

    attention_factors = positive_factors[
        positive_factors["Feature"].isin(
            [
                feature
                for feature, info in clinical_guidance.items()
                if info["type"] != "non_modifiable"
            ]
        )
    ]

    if len(attention_factors) > 0:

        attention_text = (
            "### 🎯 What deserves attention?\n\n"
        )

        for _, row in attention_factors.iterrows():

            feature = row["Feature"]

            guidance = clinical_guidance.get(
                feature
            )

            if guidance:

                attention_text += (
                    f"**{guidance['priority']} priority — "
                    f"{guidance['name']}**\n"
                    f"{guidance['guidance']}\n\n"
                )

        feedback_sections.append(attention_text)


    # --------------------------------------------------------
    # Non-modifiable context
    # --------------------------------------------------------

    context_factors = clinical_explanation[
        clinical_explanation["Feature"].isin(
            [
                feature
                for feature, info in clinical_guidance.items()
                if info["type"] == "non_modifiable"
            ]
        )
    ]

    if len(context_factors) > 0:

        context_text = (
            "### ℹ️ Non-modifiable Risk Context\n\n"
        )

        for _, row in context_factors.iterrows():

            feature = row["Feature"]

            guidance = clinical_guidance.get(
                feature
            )

            if guidance:

                context_text += (
                    f"**{guidance['name']}** — "
                    f"{guidance['guidance']}\n\n"
                )

        feedback_sections.append(context_text)


    # --------------------------------------------------------
    # Recommended next step
    # --------------------------------------------------------

    feedback_sections.append(
        "### 🩺 Recommended Next Step\n\n"
        "Because this system provides a machine-learning risk "
        "assessment, consider appropriate diabetes testing and "
        "professional medical evaluation. If you already have "
        "abnormal glucose or blood-pressure measurements, discuss "
        "them with a healthcare professional."
    )


    # --------------------------------------------------------
    # Disclaimer
    # --------------------------------------------------------

    feedback_sections.append(
        "---\n"
        "⚠️ **Important:** This system provides machine-learning "
        "risk assessment and personalized informational feedback. "
        "It is not a medical diagnosis and should not replace "
        "professional medical advice."
    )

    return "\n\n".join(feedback_sections)


# ============================================================
# CLINICAL QSVM PREDICTION FUNCTION
# ============================================================

def predict_clinical_risk(
    age,
    glucose,
    bmi,
    family_diabetes,
    hypertensive
):
    """
    Generate a diabetes-risk prediction using the finalized
    Clinical QSVM.

    IMPORTANT:
    Only the five features used during QSVM training are passed
    to the model.
    """

    # Convert Yes/No values to the original dataset encoding
    family_diabetes_value = 1 if family_diabetes == "Yes" else 0
    hypertensive_value = 1 if hypertensive == "Yes" else 0

    # Create input using EXACT training feature names
    input_data = pd.DataFrame([{
        "family_diabetes": family_diabetes_value,
        "hypertensive": hypertensive_value,
        "glucose": float(glucose),
        "bmi": float(bmi),
        "age": float(age)
    }])

    # Make sure feature order exactly matches the trained model
    input_data = input_data[clinical_features]

    # Apply the saved clinical scaler
    input_scaled = clinical_scaler.transform(input_data)

    # Get the QSVM decision score
    decision_score = float(
        clinical_model.decision_function(input_scaled)[0]
    )

    # Apply the FINAL VALIDATION-SELECTED threshold
    prediction = int(
        decision_score >= float(clinical_threshold)
    )

    # Human-readable result
    if prediction == 1:
        risk_label = "Higher Predicted Diabetes Risk"
    else:
        risk_label = "Lower Predicted Diabetes Risk"

    return {
        "prediction": prediction,
        "risk_label": risk_label,
        "decision_score": decision_score,
        "threshold": float(clinical_threshold),
        "input_data": input_data
    }




# ============================================================
# CLINICAL PATIENT-SPECIFIC SHAP ANALYSIS
# ============================================================

def analyze_clinical_shap(
    age,
    glucose,
    bmi,
    family_diabetes,
    hypertensive
):
    """
    Generate a patient-specific SHAP explanation for the
    Clinical QSVM prediction using its Random Forest surrogate.

    SHAP is used for explanation only.
    It does NOT change the QSVM prediction.
    """

    family_diabetes_value = (
        1 if family_diabetes == "Yes" else 0
    )

    hypertensive_value = (
        1 if hypertensive == "Yes" else 0
    )

    input_data = pd.DataFrame([{
        "family_diabetes": family_diabetes_value,
        "hypertensive": hypertensive_value,
        "glucose": float(glucose),
        "bmi": float(bmi),
        "age": float(age)
    }])

    # Exact model feature order
    input_data = input_data[clinical_features]

    # Same scaling used by the QSVM
    input_scaled = clinical_scaler.transform(
        input_data
    )

    # Patient-specific SHAP values
    shap_values = clinical_shap_explainer.shap_values(
        input_scaled
    )

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(
        shap_values
    ).flatten()

    explanation = pd.DataFrame({
        "Feature": clinical_features,
        "SHAP": shap_values,
        "Absolute_SHAP": np.abs(shap_values),
        "Value": input_data[
            clinical_features
        ].iloc[0].values
    })

    explanation = explanation.sort_values(
        "Absolute_SHAP",
        ascending=False
    ).reset_index(drop=True)

    # Human-readable influence
    explanation["Influence"] = explanation[
        "SHAP"
    ].apply(
        lambda x:
            "Increases risk"
            if x > 0
            else
            "Decreases risk"
            if x < 0
            else
            "No influence"
    )

    return explanation




# ============================================================
# SYMPTOM PATIENT-SPECIFIC SHAP ANALYSIS
# ============================================================

def analyze_symptom_shap(
    polyuria,
    polydipsia,
    sudden_weight_loss,
    partial_paresis,
    polyphagia,
    gender,
    irritability
):
    """
    Generate a patient-specific SHAP explanation for the
    Symptom QSVM prediction.

    SHAP is used for explanation only.
    It does NOT modify the QSVM prediction.
    """

    # --------------------------------------------------------
    # Convert inputs to original dataset encoding
    # --------------------------------------------------------

    polyuria_value = (
        1 if polyuria == "Yes" else 0
    )

    polydipsia_value = (
        1 if polydipsia == "Yes" else 0
    )

    sudden_weight_loss_value = (
        1 if sudden_weight_loss == "Yes" else 0
    )

    partial_paresis_value = (
        1 if partial_paresis == "Yes" else 0
    )

    polyphagia_value = (
        1 if polyphagia == "Yes" else 0
    )

    gender_value = (
        1 if gender == "Male" else 0
    )

    irritability_value = (
        1 if irritability == "Yes" else 0
    )

    # --------------------------------------------------------
    # Create exact QSVM input
    # --------------------------------------------------------

    input_data = pd.DataFrame([{
        "Polyuria": polyuria_value,
        "Polydipsia": polydipsia_value,
        "sudden weight loss": sudden_weight_loss_value,
        "partial paresis": partial_paresis_value,
        "Polyphagia": polyphagia_value,
        "Gender": gender_value,
        "Irritability": irritability_value
    }])

    # Exact training feature order
    input_data = input_data[symptom_features]

    # Same scaling used by Symptom QSVM
    input_scaled = symptom_scaler.transform(
        input_data
    )

    # --------------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------------

    shap_values = symptom_shap_explainer.shap_values(
        input_scaled
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(
        shap_values
    ).flatten()

    # --------------------------------------------------------
    # Build explanation table
    # --------------------------------------------------------

    explanation = pd.DataFrame({
        "Feature": symptom_features,
        "SHAP": shap_values,
        "Absolute_SHAP": np.abs(shap_values),
        "Value": input_data[
            symptom_features
        ].iloc[0].values
    })

    explanation = explanation.sort_values(
        "Absolute_SHAP",
        ascending=False
    ).reset_index(drop=True)

    explanation["Influence"] = explanation[
        "SHAP"
    ].apply(
        lambda x:
            "Increases risk"
            if x > 0
            else
            "Decreases risk"
            if x < 0
            else
            "No influence"
    )

    return explanation




# ============================================================
# DISPLAY DML CAUSAL CONTEXT
# ============================================================

def display_dml_context():

    if not dml_loaded:

        st.warning(
            "⚠️ Research causal-analysis results could not "
            "be loaded."
        )

        return

    st.divider()

    st.subheader(
        "📈 Research Causal Context"
    )

    st.caption(
        "Population-level estimates from Double Machine Learning "
        "(DML). These results provide research context and do not "
        "prove that an individual symptom or characteristic caused "
        "diabetes."
    )

    # --------------------------------------------------------
    # Convert saved results to DataFrame
    # --------------------------------------------------------

    try:

        if isinstance(dml_results, pd.DataFrame):

            dml_df = dml_results.copy()

        elif isinstance(dml_results, dict):

            dml_df = pd.DataFrame(dml_results)

        else:

            dml_df = pd.DataFrame(
                dml_results
            )

    except Exception as e:

        st.warning(
            f"Unable to format DML results: {e}"
        )

        return


    # --------------------------------------------------------
    # Normalize common column names
    # --------------------------------------------------------

    rename_map = {}

    for column in dml_df.columns:

        lower_column = str(
            column
        ).lower().replace(
            " ",
            "_"
        )

        if lower_column in [
            "feature",
            "treatment",
            "variable"
        ]:

            rename_map[column] = "Feature"

        elif lower_column in [
            "ate",
            "effect",
            "estimated_effect",
            "causal_effect"
        ]:

            rename_map[column] = "Estimated Effect"

        elif lower_column in [
            "ci_lower",
            "lower_ci",
            "lower",
            "ci_low"
        ]:

            rename_map[column] = "CI Lower"

        elif lower_column in [
            "ci_upper",
            "upper_ci",
            "upper",
            "ci_high"
        ]:

            rename_map[column] = "CI Upper"

    dml_df = dml_df.rename(
        columns=rename_map
    )


    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    display_columns = [
        column
        for column in [
            "Feature",
            "Estimated Effect",
            "CI Lower",
            "CI Upper"
        ]
        if column in dml_df.columns
    ]

    if len(display_columns) > 0:

        st.dataframe(
            dml_df[display_columns],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.dataframe(
            dml_df,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    st.info(
        """
        **How to interpret this section**

        A positive estimated effect indicates a positive association
        with the modeled diabetes outcome under the DML assumptions.
        A confidence interval that crosses zero indicates greater
        statistical uncertainty about the estimated effect.

        These estimates are based on observational data and should
        not be interpreted as proof of causation for an individual.
        """
    )


# ============================================================
# LOAD CLINICAL MODEL
# ============================================================

@st.cache_resource
def load_clinical_model():

    model = load_pickle(
        os.path.join(
            CLINICAL_DIR,
            "final_qsvm_zfeature.pkl"
        )
    )

    scaler = load_pickle(
        os.path.join(
            CLINICAL_DIR,
            "final_qsvm_scaler.pkl"
        )
    )

    features = load_pickle(
        os.path.join(
            CLINICAL_DIR,
            "final_qsvm_features.pkl"
        )
    )

    threshold = load_pickle(
        os.path.join(
            CLINICAL_DIR,
            "final_qsvm_threshold.pkl"
        )
    )

    surrogate = load_pickle(
        os.path.join(
            CLINICAL_DIR,
            "surrogate_model.pkl"
        )
    )

    return model, scaler, features, threshold, surrogate




# ============================================================
# CLINICAL SHAP EXPLAINER
# ============================================================

@st.cache_resource
def load_clinical_shap_explainer():

    surrogate_model = load_pickle(
        os.path.join(
            CLINICAL_DIR,
            "surrogate_model.pkl"
        )
    )

    # The surrogate model was trained on the same five
    # standardized QSVM features.
    explainer = shap.TreeExplainer(
        surrogate_model
    )

    return surrogate_model, explainer


# ============================================================

# ============================================================
# DML RESOURCE INITIALIZATION
# ============================================================

dml_loaded = False
dml_results = None
dml_error = None

try:
    dml_results = load_pickle(
        os.path.join(
            SYMPTOM_DIR,
            "symptom_causal_results.pkl"
        )
    )

    dml_loaded = True

except Exception as e:
    dml_loaded = False
    dml_error = str(e)

# SHAP RESOURCE INITIALIZATION
# ============================================================

# Clinical SHAP
clinical_shap_loaded = False
clinical_shap_surrogate = None
clinical_shap_explainer = None

try:

    (
        clinical_shap_surrogate,
        clinical_shap_explainer
    ) = load_clinical_shap_explainer()

    clinical_shap_loaded = True

except Exception as e:

    clinical_shap_loaded = False
    clinical_shap_error = str(e)


# Symptom SHAP
symptom_shap_loaded = False
symptom_shap_explainer = None

try:

    symptom_shap_explainer = load_pickle(
        os.path.join(
            SYMPTOM_DIR,
            "symptom_shap_explainer.pkl"
        )
    )

    symptom_shap_loaded = True

except Exception as e:

    symptom_shap_loaded = False
    symptom_shap_error = str(e)







# ============================================================
# SYMPTOM PERSONALIZED GUIDANCE
# ============================================================

symptom_guidance = {

    "Polyuria": {
        "name": "Frequent urination",
        "type": "symptom",
        "priority": "High",
        "guidance": (
            "Frequent urination can occur with elevated blood glucose. "
            "Because this symptom contributed to the risk assessment, "
            "appropriate blood-glucose testing and professional "
            "evaluation may be appropriate."
        )
    },

    "Polydipsia": {
        "name": "Excessive thirst",
        "type": "symptom",
        "priority": "High",
        "guidance": (
            "Excessive thirst can occur alongside elevated blood glucose. "
            "If this symptom is persistent, consider appropriate "
            "blood-glucose testing and discuss the result with a "
            "healthcare professional."
        )
    },

    "sudden weight loss": {
        "name": "Sudden weight loss",
        "type": "symptom",
        "priority": "High",
        "guidance": (
            "Unexplained or sudden weight loss can be associated with "
            "several health conditions, including diabetes. This symptom "
            "should not simply be suppressed and may warrant medical "
            "evaluation."
        )
    },

    "partial paresis": {
        "name": "Partial weakness or paresis",
        "type": "symptom",
        "priority": "High",
        "guidance": (
            "Partial weakness can have multiple possible causes. If it "
            "is new, persistent, or worsening, appropriate medical "
            "evaluation is recommended."
        )
    },

    "Polyphagia": {
        "name": "Increased hunger",
        "type": "symptom",
        "priority": "Moderate",
        "guidance": (
            "Increased hunger can occur for several reasons, including "
            "changes in blood-glucose regulation. Persistent unexplained "
            "changes in appetite may warrant medical evaluation."
        )
    },

    "Irritability": {
        "name": "Irritability",
        "type": "symptom",
        "priority": "Moderate",
        "guidance": (
            "Irritability can have many possible causes and is not "
            "specific to diabetes. Consider the overall symptom pattern "
            "rather than using this symptom alone to assess diabetes risk."
        )
    },

    "Gender": {
        "name": "Gender",
        "type": "non_modifiable",
        "priority": "Context",
        "guidance": (
            "Gender is a non-modifiable characteristic. It should not "
            "be treated as something the user needs to change or control."
        )
    }
}


# ============================================================
# SYMPTOM FEEDBACK GENERATOR
# ============================================================

def generate_symptom_feedback(
    prediction,
    symptom_explanation
):
    """
    Generate controlled, personalized informational feedback.

    Uses:
        - Symptom QSVM prediction
        - Patient-specific SHAP
        - Predefined symptom guidance

    Does NOT diagnose or prescribe treatment.
    """

    feedback_sections = []

    # --------------------------------------------------------
    # Risk result
    # --------------------------------------------------------

    if prediction == 1:

        feedback_sections.append(
            "🔴 **Higher Predicted Diabetes Risk**\n\n"
            "The symptom QSVM classified the current symptom "
            "profile as higher predicted diabetes risk."
        )

    else:

        feedback_sections.append(
            "🟢 **Lower Predicted Diabetes Risk**\n\n"
            "The symptom QSVM classified the current symptom "
            "profile as lower predicted diabetes risk."
        )


    # --------------------------------------------------------
    # Positive SHAP contributors
    # --------------------------------------------------------

    positive_factors = symptom_explanation[
        symptom_explanation["SHAP"] > 0
    ].copy()

    positive_factors = positive_factors.sort_values(
        "SHAP",
        ascending=False
    )

    if len(positive_factors) > 0:

        why_text = (
            "### 🧠 Why did the model make this prediction?\n\n"
        )

        for _, row in positive_factors.iterrows():

            feature = row["Feature"]
            shap_value = row["SHAP"]

            feature_name = symptom_guidance.get(
                feature,
                {}
            ).get(
                "name",
                feature
            )

            why_text += (
                f"🔴 **{feature_name}** — positive model "
                f"influence (`{shap_value:.4f}`)\n\n"
            )

        feedback_sections.append(
            why_text
        )

    else:

        feedback_sections.append(
            "### 🧠 Why did the model make this prediction?\n\n"
            "No positive SHAP contributors were identified "
            "for this patient profile."
        )


    # --------------------------------------------------------
    # What deserves attention?
    # --------------------------------------------------------

    attention_factors = positive_factors[
        positive_factors["Feature"].isin(
            [
                feature
                for feature, info in symptom_guidance.items()
                if info["type"] == "symptom"
            ]
        )
    ]

    if len(attention_factors) > 0:

        attention_text = (
            "### 🎯 What deserves attention?\n\n"
        )

        for _, row in attention_factors.iterrows():

            feature = row["Feature"]

            guidance = symptom_guidance.get(
                feature
            )

            if guidance:

                attention_text += (
                    f"**{guidance['priority']} priority — "
                    f"{guidance['name']}**\n"
                    f"{guidance['guidance']}\n\n"
                )

        feedback_sections.append(
            attention_text
        )


    # --------------------------------------------------------
    # Non-modifiable context
    # --------------------------------------------------------

    context_factors = symptom_explanation[
        symptom_explanation["Feature"].isin(
            [
                feature
                for feature, info in symptom_guidance.items()
                if info["type"] == "non_modifiable"
            ]
        )
    ]

    if len(context_factors) > 0:

        context_text = (
            "### ℹ️ Non-modifiable Context\n\n"
        )

        for _, row in context_factors.iterrows():

            feature = row["Feature"]

            guidance = symptom_guidance.get(
                feature
            )

            if guidance:

                context_text += (
                    f"**{guidance['name']}** — "
                    f"{guidance['guidance']}\n\n"
                )

        feedback_sections.append(
            context_text
        )


    # --------------------------------------------------------
    # Recommended next step
    # --------------------------------------------------------

    feedback_sections.append(
        "### 🩺 Recommended Next Step\n\n"
        "Because the model indicates a diabetes-risk pattern, "
        "consider appropriate blood-glucose testing and professional "
        "medical evaluation, particularly if relevant symptoms are "
        "persistent, unexplained, or worsening."
    )


    # --------------------------------------------------------
    # Disclaimer
    # --------------------------------------------------------

    feedback_sections.append(
        "---\n"
        "⚠️ **Important:** This system provides machine-learning "
        "risk assessment and personalized informational feedback. "
        "It is not a medical diagnosis and should not replace "
        "professional medical advice."
    )

    return "\n\n".join(
        feedback_sections
    )


# ============================================================
# SYMPTOM QSVM PREDICTION FUNCTION
# ============================================================

def predict_symptom_risk(
    polyuria,
    polydipsia,
    sudden_weight_loss,
    partial_paresis,
    polyphagia,
    gender,
    irritability
):
    """
    Generate diabetes-risk prediction using the finalized
    7-feature Symptom QSVM.

    IMPORTANT:
    Only the seven features used during QSVM training are
    passed to the model.
    """

    # --------------------------------------------------------
    # Convert Yes/No values to original dataset encoding
    # --------------------------------------------------------

    polyuria_value = (
        1 if polyuria == "Yes" else 0
    )

    polydipsia_value = (
        1 if polydipsia == "Yes" else 0
    )

    sudden_weight_loss_value = (
        1 if sudden_weight_loss == "Yes" else 0
    )

    partial_paresis_value = (
        1 if partial_paresis == "Yes" else 0
    )

    polyphagia_value = (
        1 if polyphagia == "Yes" else 0
    )

    gender_value = (
        1 if gender == "Male" else 0
    )

    irritability_value = (
        1 if irritability == "Yes" else 0
    )

    # --------------------------------------------------------
    # Create input using EXACT training feature names
    # --------------------------------------------------------

    input_data = pd.DataFrame([{
        "Polyuria": polyuria_value,
        "Polydipsia": polydipsia_value,
        "sudden weight loss": sudden_weight_loss_value,
        "partial paresis": partial_paresis_value,
        "Polyphagia": polyphagia_value,
        "Gender": gender_value,
        "Irritability": irritability_value
    }])

    # Exact training feature order
    input_data = input_data[symptom_features]

    # --------------------------------------------------------
    # Apply saved symptom scaler
    # --------------------------------------------------------

    input_scaled = symptom_scaler.transform(
        input_data
    )

    # --------------------------------------------------------
    # Get QSVM decision score
    # --------------------------------------------------------

    decision_score = float(
        symptom_model.decision_function(
            input_scaled
        )[0]
    )

    # --------------------------------------------------------
    # Apply saved validation threshold
    # --------------------------------------------------------

    prediction = int(
        decision_score >= float(symptom_threshold)
    )

    # --------------------------------------------------------
    # Human-readable result
    # --------------------------------------------------------

    if prediction == 1:

        risk_label = "Higher Predicted Diabetes Risk"

    else:

        risk_label = "Lower Predicted Diabetes Risk"

    return {
        "prediction": prediction,
        "risk_label": risk_label,
        "decision_score": decision_score,
        "threshold": float(symptom_threshold),
        "input_data": input_data
    }




# ============================================================
# SYMPTOM DML CAUSAL RESULTS
# ============================================================

@st.cache_data
def load_dml_results():

    results = load_pickle(
        os.path.join(
            SYMPTOM_DIR,
            "symptom_causal_results.pkl"
        )
    )

    return results


# ============================================================
# LOAD SYMPTOM MODEL
# ============================================================

@st.cache_resource
def load_symptom_model():

    model = load_pickle(
        os.path.join(
            SYMPTOM_DIR,
            "final_symptom_qsvm_7.pkl"
        )
    )

    scaler = load_pickle(
        os.path.join(
            SYMPTOM_DIR,
            "final_symptom_scaler_7.pkl"
        )
    )

    features = load_pickle(
        os.path.join(
            SYMPTOM_DIR,
            "final_symptom_features_7.pkl"
        )
    )

    threshold = load_pickle(
        os.path.join(
            SYMPTOM_DIR,
            "final_symptom_threshold_7.pkl"
        )
    )

    surrogate = load_pickle(
        os.path.join(
            SYMPTOM_DIR,
            "symptom_shap_surrogate.pkl"
        )
    )

    return model, scaler, features, threshold, surrogate


# ============================================================
# LOAD MODELS
# ============================================================

try:

    (
        clinical_model,
        clinical_scaler,
        clinical_features,
        clinical_threshold,
        clinical_surrogate
    ) = load_clinical_model()

    clinical_loaded = True

except Exception as e:

    clinical_loaded = False
    clinical_error = str(e)


try:

    (
        symptom_model,
        symptom_scaler,
        symptom_features,
        symptom_threshold,
        symptom_surrogate
    ) = load_symptom_model()

    symptom_loaded = True

except Exception as e:

    symptom_loaded = False
    symptom_error = str(e)


# ============================================================
# HEADER
# ============================================================

st.title("🩺 Diabetes Risk Analysis System")

st.markdown(
    """
    **QSVM-based diabetes risk assessment with personalized
    model explanations.**
    
    Select an assessment method below to begin.
    """
)

st.divider()


# ============================================================
# ASSESSMENT METHOD
# ============================================================

assessment_method = st.radio(
    "Choose Assessment Method",
    [
        "🩺 Clinical Data",
        "🧠 Symptoms"
    ],
    horizontal=True
)


# ============================================================
# CLINICAL DATA MODE
# ============================================================

if assessment_method == "🩺 Clinical Data":

    st.header("🩺 Clinical Data Assessment")

    st.info(
        "⭐ Fields marked as required are used directly by the "
        "Clinical QSVM model. Additional health information is "
        "collected for context and does not affect the QSVM prediction."
    )

    # --------------------------------------------------------
    # REQUIRED MODEL FEATURES
    # --------------------------------------------------------

    st.subheader("⭐ Required Model Features")

    st.caption(
        "These 5 features are used by the finalized Clinical QSVM."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age *",
            min_value=1,
            max_value=120,
            value=30,
            step=1
        )

    with col2:

        glucose = st.number_input(
            "Blood Glucose *",
            min_value=0.1,
            max_value=50.0,
            value=5.5,
            step=0.1,
            format="%.1f"
        )

    with col3:

        bmi = st.number_input(
            "BMI *",
            min_value=10.0,
            max_value=60.0,
            value=22.0,
            step=0.1,
            format="%.1f"
        )

    col1, col2 = st.columns(2)

    with col1:

        family_diabetes = st.selectbox(
            "Family History of Diabetes *",
            ["No", "Yes"]
        )

    with col2:

        hypertensive = st.selectbox(
            "Hypertension *",
            ["No", "Yes"]
        )

    # --------------------------------------------------------
    # ADDITIONAL CLINICAL INFORMATION
    # --------------------------------------------------------

    st.subheader("➕ Additional Health Information")

    st.caption(
        "These fields provide additional patient context. "
        "They are NOT used as inputs to the Clinical QSVM."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

    with col2:

        pulse_rate = st.number_input(
            "Pulse Rate (bpm)",
            min_value=30,
            max_value=200,
            value=72,
            step=1
        )

    with col3:

        systolic_bp = st.number_input(
            "Systolic Blood Pressure (mmHg)",
            min_value=50,
            max_value=250,
            value=120,
            step=1
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        diastolic_bp = st.number_input(
            "Diastolic Blood Pressure (mmHg)",
            min_value=30,
            max_value=150,
            value=80,
            step=1
        )

    with col2:

        height = st.number_input(
            "Height (m)",
            min_value=1.2,
            max_value=2.2,
            value=1.70,
            step=0.01,
            format="%.2f"
        )

    with col3:

        weight = st.number_input(
            "Weight (kg)",
            min_value=30.0,
            max_value=150.0,
            value=65.0,
            step=0.1,
            format="%.1f"
        )

    # --------------------------------------------------------
    # INFORMATION PANEL
    # --------------------------------------------------------

    st.divider()

    st.subheader("🔬 Model Input Transparency")

    st.write(
        "The Clinical QSVM was trained using exactly these "
        "five features:"
    )

    st.code(
        ", ".join(clinical_features),
        language="text"
    )

    st.caption(
        "Additional clinical fields displayed above are collected "
        "for contextual information only and are not passed to the QSVM."
    )

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Assess Diabetes Risk",
        type="primary",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # Run Clinical QSVM
        # ----------------------------------------------------

        if not clinical_loaded:

            st.error(
                "❌ Clinical QSVM could not be loaded. "
                "Please check the model files and dependencies."
            )

        else:

            try:

                clinical_result = predict_clinical_risk(
                    age=age,
                    glucose=glucose,
                    bmi=bmi,
                    family_diabetes=family_diabetes,
                    hypertensive=hypertensive
                )

                st.divider()

                st.subheader("📊 Clinical QSVM Assessment Result")

                # ------------------------------------------------
                # Risk result
                # ------------------------------------------------

                if clinical_result["prediction"] == 1:

                    st.error(
                        "🔴 " + clinical_result["risk_label"]
                    )

                else:

                    st.success(
                        "🟢 " + clinical_result["risk_label"]
                    )

                # ------------------------------------------------
                # Decision score
                # ------------------------------------------------

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "QSVM Decision Score",
                        f"{clinical_result['decision_score']:.4f}"
                    )

                with col2:

                    st.metric(
                        "Model Threshold",
                        f"{clinical_result['threshold']:.4f}"
                    )

                # ------------------------------------------------
                # Model inputs
                # ------------------------------------------------

                with st.expander(
                    "🔬 View Features Used by Clinical QSVM"
                ):

                    st.dataframe(
                        clinical_result["input_data"],
                        use_container_width=True,
                        hide_index=True
                    )

                st.info(
                    "The result above is generated using only the "
                    "five clinical features used during QSVM training. "
                    "Additional health information collected in this "
                    "form does not influence the prediction."
                )

                # ------------------------------------------------
                # PATIENT-SPECIFIC SHAP EXPLANATION
                # ------------------------------------------------

                if clinical_shap_loaded:

                    try:

                        clinical_explanation = analyze_clinical_shap(
                            age=age,
                            glucose=glucose,
                            bmi=bmi,
                            family_diabetes=family_diabetes,
                            hypertensive=hypertensive
                        )

                        st.divider()

                        st.subheader(
                            "🧠 Why did the model make this prediction?"
                        )

                        st.caption(
                            "Patient-specific SHAP explanation based "
                            "on a surrogate model of the Clinical QSVM."
                        )

                        # Positive contributors
                        positive_factors = clinical_explanation[
                            clinical_explanation["SHAP"] > 0
                        ]

                        negative_factors = clinical_explanation[
                            clinical_explanation["SHAP"] < 0
                        ]

                        if len(positive_factors) > 0:

                            st.markdown(
                                "#### 🔴 Factors increasing predicted risk"
                            )

                            for _, row in positive_factors.iterrows():

                                st.write(
                                    f"**{row['Feature']}** — "
                                    f"positive model influence "
                                    f"(`{row['SHAP']:.4f}`)"
                                )

                        if len(negative_factors) > 0:

                            with st.expander(
                                "🟢 Factors decreasing predicted risk"
                            ):

                                for _, row in negative_factors.iterrows():

                                    st.write(
                                        f"**{row['Feature']}** — "
                                        f"negative model influence "
                                        f"(`{row['SHAP']:.4f}`)"
                                    )

                        # Full explanation
                        with st.expander(
                            "🔬 View Advanced SHAP Explanation"
                        ):

                            st.dataframe(
                                clinical_explanation[
                                    [
                                        "Feature",
                                        "Value",
                                        "SHAP",
                                        "Influence"
                                    ]
                                ],
                                use_container_width=True,
                                hide_index=True
                            )

                            st.caption(
                                "SHAP values explain the surrogate "
                                "model's approximation of the QSVM "
                                "decision function. They are not "
                                "causal effects."
                            )

                    except Exception as e:

                        st.warning(
                            "⚠️ Clinical SHAP explanation could not "
                            f"be generated: {e}"
                        )

                else:

                    st.warning(
                        "⚠️ Clinical SHAP explainer could not be loaded."
                    )

                # ------------------------------------------------
                # PERSONALIZED CLINICAL FEEDBACK
                # ------------------------------------------------

                if clinical_shap_loaded:

                    try:

                        clinical_feedback = (
                            generate_clinical_feedback(
                                prediction=clinical_result["prediction"],
                                clinical_explanation=clinical_explanation
                            )
                        )

                        st.divider()

                        st.subheader(
                            "🩺 Personalized Health Feedback"
                        )

                        st.markdown(
                            clinical_feedback
                        )

                    except Exception as e:

                        st.warning(
                            "⚠️ Personalized clinical feedback "
                            f"could not be generated: {e}"
                        )

                # ------------------------------------------------
                # PERSONALIZED SYMPTOM FEEDBACK
                # ------------------------------------------------

                if symptom_shap_loaded:

                    try:

                        symptom_feedback = (
                            generate_symptom_feedback(
                                prediction=symptom_result["prediction"],
                                symptom_explanation=symptom_explanation
                            )
                        )

                        st.divider()

                        st.subheader(
                            "🩺 Personalized Health Feedback"
                        )

                        st.markdown(
                            symptom_feedback
                        )

                    except Exception as e:

                        st.warning(
                            "⚠️ Personalized symptom feedback "
                            f"could not be generated: {e}"
                        )

                st.warning(
                    "⚠️ This is a machine-learning risk assessment, "
                    "not a medical diagnosis."
                )

            except Exception as e:

                st.error(
                    f"❌ Error during Clinical QSVM prediction:\n\n{e}"
                )


# ============================================================
# SYMPTOM MODE
# ============================================================

else:

    st.header("🧠 Symptom-Based Assessment")

    st.info(
        "⭐ Fields marked as required are used directly by the "
        "Symptom QSVM model. Additional symptoms are collected "
        "for context and do not affect the QSVM prediction."
    )

    # --------------------------------------------------------
    # REQUIRED MODEL FEATURES
    # --------------------------------------------------------

    st.subheader("⭐ Required Model Features")

    st.caption(
        "These 7 features are used by the finalized Symptom QSVM."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        polyuria = st.selectbox(
            "Frequent Urination (Polyuria) *",
            ["No", "Yes"]
        )

    with col2:

        polydipsia = st.selectbox(
            "Excessive Thirst (Polydipsia) *",
            ["No", "Yes"]
        )

    with col3:

        sudden_weight_loss = st.selectbox(
            "Sudden Weight Loss *",
            ["No", "Yes"]
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        partial_paresis = st.selectbox(
            "Partial Weakness / Paresis *",
            ["No", "Yes"]
        )

    with col2:

        polyphagia = st.selectbox(
            "Increased Hunger (Polyphagia) *",
            ["No", "Yes"]
        )

    with col3:

        gender = st.selectbox(
            "Gender *",
            ["Female", "Male"]
        )

    irritability = st.selectbox(
        "Irritability *",
        ["No", "Yes"]
    )

    # --------------------------------------------------------
    # ADDITIONAL SYMPTOM INFORMATION
    # --------------------------------------------------------

    st.subheader("➕ Additional Symptom Information")

    st.caption(
        "These symptoms are collected for additional context. "
        "They are NOT used as inputs to the Symptom QSVM."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        weakness = st.selectbox(
            "Weakness",
            ["No", "Yes"]
        )

    with col2:

        genital_thrush = st.selectbox(
            "Genital Thrush",
            ["No", "Yes"]
        )

    with col3:

        visual_blurring = st.selectbox(
            "Visual Blurring",
            ["No", "Yes"]
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        itching = st.selectbox(
            "Itching",
            ["No", "Yes"]
        )

    with col2:

        delayed_healing = st.selectbox(
            "Delayed Healing",
            ["No", "Yes"]
        )

    with col3:

        muscle_stiffness = st.selectbox(
            "Muscle Stiffness",
            ["No", "Yes"]
        )

    col1, col2 = st.columns(2)

    with col1:

        alopecia = st.selectbox(
            "Alopecia",
            ["No", "Yes"]
        )

    with col2:

        obesity = st.selectbox(
            "Obesity",
            ["No", "Yes"]
        )

    # --------------------------------------------------------
    # INFORMATION PANEL
    # --------------------------------------------------------

    st.divider()

    st.subheader("🔬 Model Input Transparency")

    st.write(
        "The Symptom QSVM was trained using exactly these "
        "seven features:"
    )

    st.code(
        ", ".join(symptom_features),
        language="text"
    )

    st.caption(
        "Additional symptom fields displayed above are collected "
        "for contextual information only and are not passed to the QSVM."
    )

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Assess Diabetes Risk",
        type="primary",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # Run Symptom QSVM
        # ----------------------------------------------------

        if not symptom_loaded:

            st.error(
                "❌ Symptom QSVM could not be loaded. "
                "Please check the model files and dependencies."
            )

        else:

            try:

                symptom_result = predict_symptom_risk(
                    polyuria=polyuria,
                    polydipsia=polydipsia,
                    sudden_weight_loss=sudden_weight_loss,
                    partial_paresis=partial_paresis,
                    polyphagia=polyphagia,
                    gender=gender,
                    irritability=irritability
                )

                st.divider()

                st.subheader(
                    "📊 Symptom QSVM Assessment Result"
                )

                # ------------------------------------------------
                # Risk result
                # ------------------------------------------------

                if symptom_result["prediction"] == 1:

                    st.error(
                        "🔴 " + symptom_result["risk_label"]
                    )

                else:

                    st.success(
                        "🟢 " + symptom_result["risk_label"]
                    )

                # ------------------------------------------------
                # Decision score and threshold
                # ------------------------------------------------

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "QSVM Decision Score",
                        f"{symptom_result['decision_score']:.4f}"
                    )

                with col2:

                    st.metric(
                        "Model Threshold",
                        f"{symptom_result['threshold']:.4f}"
                    )

                # ------------------------------------------------
                # Model inputs
                # ------------------------------------------------

                with st.expander(
                    "🔬 View Features Used by Symptom QSVM"
                ):

                    st.dataframe(
                        symptom_result["input_data"],
                        use_container_width=True,
                        hide_index=True
                    )

                st.info(
                    "The result above is generated using only the "
                    "seven symptom features used during QSVM training. "
                    "Additional symptom information collected in this "
                    "form does not influence the prediction."
                )

                # ------------------------------------------------
                # PATIENT-SPECIFIC SHAP EXPLANATION
                # ------------------------------------------------

                if symptom_shap_loaded:

                    try:

                        symptom_explanation = (
                            analyze_symptom_shap(
                                polyuria=polyuria,
                                polydipsia=polydipsia,
                                sudden_weight_loss=sudden_weight_loss,
                                partial_paresis=partial_paresis,
                                polyphagia=polyphagia,
                                gender=gender,
                                irritability=irritability
                            )
                        )

                        st.divider()

                        st.subheader(
                            "🧠 Why did the model make this prediction?"
                        )

                        st.caption(
                            "Patient-specific SHAP explanation "
                            "based on the saved Symptom QSVM "
                            "surrogate explainer."
                        )

                        # Positive contributors
                        positive_factors = symptom_explanation[
                            symptom_explanation["SHAP"] > 0
                        ]

                        # Negative contributors
                        negative_factors = symptom_explanation[
                            symptom_explanation["SHAP"] < 0
                        ]

                        if len(positive_factors) > 0:

                            st.markdown(
                                "#### 🔴 Factors increasing predicted risk"
                            )

                            for _, row in positive_factors.iterrows():

                                st.write(
                                    f"**{row['Feature']}** — "
                                    f"positive model influence "
                                    f"(`{row['SHAP']:.4f}`)"
                                )

                        if len(negative_factors) > 0:

                            with st.expander(
                                "🟢 Factors decreasing predicted risk"
                            ):

                                for _, row in negative_factors.iterrows():

                                    st.write(
                                        f"**{row['Feature']}** — "
                                        f"negative model influence "
                                        f"(`{row['SHAP']:.4f}`)"
                                    )

                        # Full explanation
                        with st.expander(
                            "🔬 View Advanced SHAP Explanation"
                        ):

                            st.dataframe(
                                symptom_explanation[
                                    [
                                        "Feature",
                                        "Value",
                                        "SHAP",
                                        "Influence"
                                    ]
                                ],
                                use_container_width=True,
                                hide_index=True
                            )

                            st.caption(
                                "SHAP values explain the surrogate "
                                "model's approximation of the QSVM "
                                "decision function. They are not "
                                "causal effects."
                            )

                    except Exception as e:

                        st.warning(
                            "⚠️ Symptom SHAP explanation could not "
                            f"be generated: {e}"
                        )

                else:

                    st.warning(
                        "⚠️ Symptom SHAP explainer could not be loaded."
                    )



                # ------------------------------------------------
                # RESEARCH DML CONTEXT
                # ------------------------------------------------

                display_dml_context()


                st.warning(
                    "⚠️ This is a machine-learning risk assessment, "
                    "not a medical diagnosis."
                )

            except Exception as e:

                st.error(
                    f"❌ Error during Symptom QSVM prediction:\n\n{e}"
                )
