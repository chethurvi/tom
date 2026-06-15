 import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Employee Attrition Prediction Using Machine Learning")
st.write("""
This Streamlit application predicts whether an employee is likely to leave the organisation.
It uses the IBM HR Analytics Employee Attrition dataset and applies machine learning models.
""")

@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset file not found. Please place `WA_Fn-UseC_-HR-Employee-Attrition.csv` in the same folder as app.py.")
    st.stop()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Project Overview", "Dataset", "EDA", "Model Training", "Prediction"]
)

if page == "Project Overview":
    st.header("Project Overview")

    st.write("""
    The aim of this project is to predict employee attrition using machine learning.
    Employee attrition means employees leaving the organisation. Predicting attrition helps
    HR departments identify employees who may be at risk of leaving and take suitable action.
    """)

    st.subheader("Project Pipeline")
    st.markdown("""
    - Data Understanding
    - Data Preprocessing
    - Exploratory Data Analysis
    - Machine Learning Model Training
    - Model Evaluation
    - Employee Attrition Prediction
    """)

    st.subheader("Dataset Information")
    st.write("""
    The dataset used is the IBM HR Analytics Employee Attrition dataset.
    It contains employee demographic, job-related and income-related information.
    """)

elif page == "Dataset":
    st.header("Dataset Preview")

    st.subheader("First Five Rows")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("Data Types")
    st.dataframe(df.dtypes.astype(str))

elif page == "EDA":
    st.header("Exploratory Data Analysis")

    st.subheader("Attrition Distribution")
    attrition_count = df["Attrition"].value_counts().reset_index()
    attrition_count.columns = ["Attrition", "Count"]

    fig = px.bar(
        attrition_count,
        x="Attrition",
        y="Count",
        title="Employee Attrition Count",
        text="Count"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Attrition by Department")
    fig = px.histogram(
        df,
        x="Department",
        color="Attrition",
        barmode="group",
        title="Attrition by Department"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Income by Attrition")
    fig = px.box(
        df,
        x="Attrition",
        y="MonthlyIncome",
        color="Attrition",
        title="Monthly Income and Attrition"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Age Distribution")
    fig = px.histogram(
        df,
        x="Age",
        color="Attrition",
        nbins=30,
        title="Age Distribution by Attrition"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Job Satisfaction and Attrition")
    fig = px.histogram(
        df,
        x="JobSatisfaction",
        color="Attrition",
        barmode="group",
        title="Job Satisfaction and Attrition"
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Model Training":
    st.header("Machine Learning Model Training")

    data = df.copy()

    label_encoders = {}
    for column in data.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    X = data.drop("Attrition", axis=1)
    y = data["Attrition"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model_choice = st.selectbox(
        "Select Machine Learning Model",
        ["Random Forest", "Logistic Regression"]
    )

    if model_choice == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Model Accuracy")
    st.success(f"{model_choice} Accuracy: {accuracy:.2f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(cm, columns=["Predicted No", "Predicted Yes"], index=["Actual No", "Actual Yes"]))

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    if model_choice == "Random Forest":
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig = px.bar(
            feature_importance.head(10),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Important Features"
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Prediction":
    st.header("Employee Attrition Prediction")

    data = df.copy()

    label_encoders = {}
    for column in data.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    X = data.drop("Attrition", axis=1)
    y = data["Attrition"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_scaled, y)

    st.write("Enter employee details below to predict attrition risk.")

    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.slider("Age", 18, 60, 30)
        DailyRate = st.number_input("Daily Rate", 100, 1500, 800)
        DistanceFromHome = st.slider("Distance From Home", 1, 30, 5)
        Education = st.selectbox("Education", [1, 2, 3, 4, 5])
        EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
        HourlyRate = st.number_input("Hourly Rate", 30, 100, 60)
        JobInvolvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
        JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])

    with col2:
        JobSatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)
        MonthlyRate = st.number_input("Monthly Rate", 2000, 27000, 10000)
        NumCompaniesWorked = st.slider("Number of Companies Worked", 0, 9, 1)
        PercentSalaryHike = st.slider("Percent Salary Hike", 10, 25, 15)
        PerformanceRating = st.selectbox("Performance Rating", [3, 4])
        RelationshipSatisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
        StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])

    with col3:
        TotalWorkingYears = st.slider("Total Working Years", 0, 40, 5)
        TrainingTimesLastYear = st.slider("Training Times Last Year", 0, 6, 2)
        WorkLifeBalance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
        YearsAtCompany = st.slider("Years At Company", 0, 40, 3)
        YearsInCurrentRole = st.slider("Years In Current Role", 0, 18, 2)
        YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 1)
        YearsWithCurrManager = st.slider("Years With Current Manager", 0, 17, 2)

    BusinessTravel = st.selectbox("Business Travel", df["BusinessTravel"].unique())
    Department = st.selectbox("Department", df["Department"].unique())
    EducationField = st.selectbox("Education Field", df["EducationField"].unique())
    Gender = st.selectbox("Gender", df["Gender"].unique())
    JobRole = st.selectbox("Job Role", df["JobRole"].unique())
    MaritalStatus = st.selectbox("Marital Status", df["MaritalStatus"].unique())
    OverTime = st.selectbox("Over Time", df["OverTime"].unique())

    input_data = pd.DataFrame({
        "Age": [Age],
        "BusinessTravel": [BusinessTravel],
        "DailyRate": [DailyRate],
        "Department": [Department],
        "DistanceFromHome": [DistanceFromHome],
        "Education": [Education],
        "EducationField": [EducationField],
        "EmployeeCount": [1],
        "EmployeeNumber": [9999],
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
        "Over18": ["Y"],
        "OverTime": [OverTime],
        "PercentSalaryHike": [PercentSalaryHike],
        "PerformanceRating": [PerformanceRating],
        "RelationshipSatisfaction": [RelationshipSatisfaction],
        "StandardHours": [80],
        "StockOptionLevel": [StockOptionLevel],
        "TotalWorkingYears": [TotalWorkingYears],
        "TrainingTimesLastYear": [TrainingTimesLastYear],
        "WorkLifeBalance": [WorkLifeBalance],
        "YearsAtCompany": [YearsAtCompany],
        "YearsInCurrentRole": [YearsInCurrentRole],
        "YearsSinceLastPromotion": [YearsSinceLastPromotion],
        "YearsWithCurrManager": [YearsWithCurrManager]
    })

    for column in input_data.select_dtypes(include=["object"]).columns:
        if column in label_encoders:
            input_data[column] = label_encoders[column].transform(input_data[column])

    input_data = input_data[X.columns]

    input_scaled = scaler.transform(input_data)

    if st.button("Predict Attrition"):
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)

        if prediction[0] == 1:
            st.error("Prediction: Employee is likely to leave the company.")
        else:
            st.success("Prediction: Employee is likely to stay in the company.")

        st.write(f"Probability of Staying: {probability[0][0]:.2f}")
        st.write(f"Probability of Leaving: {probability[0][1]:.2f}")

st.sidebar.markdown("---")
st.sidebar.write("Developed for Advanced Data Science Project")
