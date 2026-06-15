import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from imblearn.over_sampling import SMOTE


st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Employee Attrition Prediction Using Machine Learning")
st.write(
    "This Streamlit application predicts employee attrition using the IBM HR Analytics dataset. "
    "It includes preprocessing, EDA, SMOTE balancing, model evaluation and employee-level prediction."
)


@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")


try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset file not found. Please upload `WA_Fn-UseC_-HR-Employee-Attrition.csv` in the same folder as app.py.")
    st.stop()


@st.cache_data
def prepare_data(df):
    data = df.copy()

    drop_cols = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    data = data.drop(columns=drop_cols)

    data["Attrition"] = data["Attrition"].map({"Yes": 1, "No": 0})

    label_encoders = {}
    for col in data.select_dtypes(include="object").columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    X = data.drop("Attrition", axis=1)
    y = data["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    return X, y, X_train_smote, X_test_scaled, y_train_smote, y_test, scaler, label_encoders


X, y, X_train, X_test, y_train, y_test, scaler, label_encoders = prepare_data(df)


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Project Overview",
        "Dataset Overview",
        "Exploratory Data Analysis",
        "Model Training and Evaluation",
        "Employee Attrition Predictor"
    ]
)


if page == "Project Overview":
    st.header("Project Overview")

    st.write("""
    Employee attrition means employees leaving an organisation. This project applies machine learning
    to predict whether an employee is likely to leave based on demographic, job-related, income-related
    and satisfaction-related features.
    """)

    st.subheader("Project Pipeline")
    st.markdown("""
    - Dataset loading and understanding
    - Data preprocessing
    - Categorical encoding
    - Feature scaling
    - Class imbalance handling using SMOTE
    - Exploratory Data Analysis
    - Logistic Regression and Random Forest modelling
    - Model evaluation using Accuracy, Precision, Recall, F1-score and ROC-AUC
    - Streamlit-based employee attrition prediction
    """)

    st.subheader("Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Records", df.shape[0])
    col2.metric("Features", df.shape[1])
    col3.metric("Target", "Attrition")

    st.subheader("Why this project is useful")
    st.write("""
    The application can support HR decision-making by identifying employees with higher attrition risk.
    It should be used as a decision-support tool rather than a replacement for human judgement.
    """)


elif page == "Dataset Overview":
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("First Five Rows")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Dataset Information")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })
    st.dataframe(info_df, use_container_width=True)

    st.subheader("Attrition Class Distribution")
    attrition_counts = df["Attrition"].value_counts().reset_index()
    attrition_counts.columns = ["Attrition", "Count"]
    attrition_counts["Percentage"] = round(attrition_counts["Count"] / len(df) * 100, 2)

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(attrition_counts, use_container_width=True)

    with col2:
        fig = px.pie(
            attrition_counts,
            names="Attrition",
            values="Count",
            title="Attrition Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Class Imbalance Note")
    st.info(
        "The dataset is moderately imbalanced. Most employees are in the 'No Attrition' class, "
        "while a smaller proportion are in the 'Attrition' class. Therefore, SMOTE is applied "
        "to the training data during model development."
    )


elif page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Attrition Distribution")
        attrition_count = df["Attrition"].value_counts().reset_index()
        attrition_count.columns = ["Attrition", "Count"]

        fig = px.bar(
            attrition_count,
            x="Attrition",
            y="Count",
            text="Count",
            title="Employee Attrition Count"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Attrition by Overtime")
        fig = px.histogram(
            df,
            x="OverTime",
            color="Attrition",
            barmode="group",
            title="Attrition by Overtime"
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Attrition by Department")
        fig = px.histogram(
            df,
            x="Department",
            color="Attrition",
            barmode="group",
            title="Attrition by Department"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Monthly Income by Attrition")
        fig = px.box(
            df,
            x="Attrition",
            y="MonthlyIncome",
            color="Attrition",
            title="Monthly Income and Attrition"
        )
        st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Age Distribution by Attrition")
        fig = px.histogram(
            df,
            x="Age",
            color="Attrition",
            nbins=30,
            title="Age Distribution by Attrition"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.subheader("Job Satisfaction and Attrition")
        fig = px.histogram(
            df,
            x="JobSatisfaction",
            color="Attrition",
            barmode="group",
            title="Job Satisfaction and Attrition"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        title="Correlation Heatmap of Numerical Features",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)


elif page == "Model Training and Evaluation":
    st.header("Model Training and Evaluation")

    st.write(
        "The models are trained using SMOTE-balanced training data and evaluated on the original test data."
    )

    st.subheader("SMOTE Class Balancing")

    before_smote = pd.Series(y).value_counts().reset_index()
    before_smote.columns = ["Attrition Class", "Count"]
    before_smote["Stage"] = "Before SMOTE"

    after_smote = pd.Series(y_train).value_counts().reset_index()
    after_smote.columns = ["Attrition Class", "Count"]
    after_smote["Stage"] = "After SMOTE"

    smote_df = pd.concat([before_smote, after_smote], ignore_index=True)
    smote_df["Attrition Class"] = smote_df["Attrition Class"].map({0: "No", 1: "Yes"})

    fig = px.bar(
        smote_df,
        x="Attrition Class",
        y="Count",
        color="Stage",
        barmode="group",
        title="Class Distribution Before and After SMOTE"
    )
    st.plotly_chart(fig, use_container_width=True)

    model_choice = st.selectbox(
        "Select Machine Learning Model",
        ["Logistic Regression", "Random Forest"]
    )

    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    st.subheader("Model Performance Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 3))
    col2.metric("Precision", round(precision_score(y_test, y_pred), 3))
    col3.metric("Recall", round(recall_score(y_test, y_pred), 3))
    col4.metric("F1-score", round(f1_score(y_test, y_pred), 3))
    col5.metric("ROC-AUC", round(roc_auc_score(y_test, y_prob), 3))

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"]
    )

    st.dataframe(cm_df, use_container_width=True)

    fig = px.imshow(
        cm_df,
        text_auto=True,
        title="Confusion Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

    if model_choice == "Random Forest":
        st.subheader("Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(15)

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Important Features"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(importance_df, use_container_width=True)


elif page == "Employee Attrition Predictor":
    st.header("Employee Attrition Predictor")

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    st.write("Enter employee details below to predict attrition risk.")

    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.slider("Age", 18, 60, 30)
        BusinessTravel = st.selectbox("Business Travel", df["BusinessTravel"].unique())
        Department = st.selectbox("Department", df["Department"].unique())
        DistanceFromHome = st.slider("Distance From Home", 1, 30, 5)
        Education = st.selectbox("Education", [1, 2, 3, 4, 5])
        EducationField = st.selectbox("Education Field", df["EducationField"].unique())
        EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
        Gender = st.selectbox("Gender", df["Gender"].unique())
        JobInvolvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
        JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])

    with col2:
        JobRole = st.selectbox("Job Role", df["JobRole"].unique())
        JobSatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        MaritalStatus = st.selectbox("Marital Status", df["MaritalStatus"].unique())
        MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)
        MonthlyRate = st.number_input("Monthly Rate", 2000, 27000, 10000)
        NumCompaniesWorked = st.slider("Number of Companies Worked", 0, 9, 1)
        OverTime = st.selectbox("Over Time", df["OverTime"].unique())
        PercentSalaryHike = st.slider("Percent Salary Hike", 10, 25, 15)
        PerformanceRating = st.selectbox("Performance Rating", [3, 4])
        RelationshipSatisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])

    with col3:
        StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        TotalWorkingYears = st.slider("Total Working Years", 0, 40, 5)
        TrainingTimesLastYear = st.slider("Training Times Last Year", 0, 6, 2)
        WorkLifeBalance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
        YearsAtCompany = st.slider("Years At Company", 0, 40, 3)
        YearsInCurrentRole = st.slider("Years In Current Role", 0, 18, 2)
        YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 1)
        YearsWithCurrManager = st.slider("Years With Current Manager", 0, 17, 2)
        DailyRate = st.number_input("Daily Rate", 100, 1500, 800)
        HourlyRate = st.number_input("Hourly Rate", 30, 100, 60)

    input_data = pd.DataFrame({
        "Age": [Age],
        "BusinessTravel": [BusinessTravel],
        "DailyRate": [DailyRate],
        "Department": [Department],
        "DistanceFromHome": [DistanceFromHome],
        "Education": [Education],
        "EducationField": [EducationField],
        "EnvironmentSatisfaction": [EnvironmentSatisfaction],
        "Gender": [Gender],
        "HourlyRate": [HourlyRate],
        "JobInvolvement": [JobInvolvement],
        "JobLevel": [JobLevel],
        "JobRole": [JobRole],
        "JobSatisfaction": [JobSatisfaction],
        "MaritalStatus": [MaritalStatus],
        "MonthlyIncome": [MonthlyIncome],
        "MonthlyRate": [MonthlyRate],
        "NumCompaniesWorked": [NumCompaniesWorked],
        "OverTime": [OverTime],
        "PercentSalaryHike": [PercentSalaryHike],
        "PerformanceRating": [PerformanceRating],
        "RelationshipSatisfaction": [RelationshipSatisfaction],
        "StockOptionLevel": [StockOptionLevel],
        "TotalWorkingYears": [TotalWorkingYears],
        "TrainingTimesLastYear": [TrainingTimesLastYear],
        "WorkLifeBalance": [WorkLifeBalance],
        "YearsAtCompany": [YearsAtCompany],
        "YearsInCurrentRole": [YearsInCurrentRole],
        "YearsSinceLastPromotion": [YearsSinceLastPromotion],
        "YearsWithCurrManager": [YearsWithCurrManager]
    })

    for col in input_data.select_dtypes(include="object").columns:
        input_data[col] = label_encoders[col].transform(input_data[col])

    input_data = input_data[X.columns]
    input_scaled = scaler.transform(input_data)

    if st.button("Predict Attrition"):
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Prediction: Employee is likely to leave the company.")
        else:
            st.success("Prediction: Employee is likely to stay in the company.")

        st.write(f"Probability of Leaving: **{probability:.2f}**")
        st.write(f"Probability of Staying: **{1 - probability:.2f}**")

        if probability < 0.30:
            st.success("Risk Level: Low")
        elif probability < 0.60:
            st.warning("Risk Level: Medium")
        else:
            st.error("Risk Level: High")


st.sidebar.markdown("---")
st.sidebar.write("Developed for COMP11068 Advanced Data Science Project")
