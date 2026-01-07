# Predicting Customer Churn  
**Identifying Susceptible Customers in Telecommunications**

## Project Overview
This project focuses on predicting customer churn for a telecommunications company, Reder Telecom. In a highly competitive market, customer retention is critical for revenue protection, marketing efficiency, and long-term customer satisfaction.

The goal is to identify customers who are most likely to churn, enabling proactive retention strategies and more targeted engagement efforts.

---

## Business Context
Reder Telecom operates in a competitive telecommunications environment and faces increasing customer churn due to several pressures:

- Intense competition offering similar services  
- Pricing pressure affecting profitability  
- Network quality and performance issues  
- Evolving customer expectations and preferences  
- Challenges in building long-term customer loyalty  

Understanding and predicting churn is essential for addressing these challenges and improving customer retention.

---

## Why Churn Prediction Matters
Customer churn has a direct impact on both revenue and operational efficiency. Predictive churn analysis supports the business by enabling:

- **Cost reduction**, as retaining existing customers is more cost-effective than acquiring new ones  
- **Revenue growth**, since retained customers are more likely to make repeat or additional purchases  
- **Improved customer satisfaction**, through better understanding of customer needs  
- **Competitive advantage**, by acting proactively rather than reactively  
- **Data-driven decision-making**, replacing intuition with analytical insight  

---

## Project Objectives
This project is structured around the following objectives:

- Develop a classification model to predict customer churn using historical customer data  
- Perform data preprocessing, feature engineering, and exploratory data analysis  
- Train and evaluate multiple classification models  
- Compare model performance using appropriate evaluation metrics  
- Build an interactive application to demonstrate churn predictions  

---

## Data Overview
The dataset used in this project includes multiple aspects of customer information:

- **Customer Demographics**
  - Identification and profile attributes  

- **Purchase History**
  - Transaction and billing details  

- **Subscription Details**
  - Service plans and usage information  

- **Customer Engagement and Feedback**
  - Interaction history and service feedback  

- **Marketing and Churn Labels**
  - Campaign exposure and churn outcomes  

---

## Modelling Approach
The project applies supervised machine learning techniques for churn prediction.

Key aspects of the modelling process include:
- Handling class imbalance in churn labels  
- Feature selection and preprocessing  
- Training and evaluating multiple classification algorithms  
- Comparing models using standard performance metrics  

Planned and evaluated models include:
- Logistic Regression  
- Random Forest  

---

## Deployment and Application
An interactive application is developed to demonstrate churn predictions and support exploration by non-technical users.

- **Frontend**: Streamlit interface for user interaction  
- **Deployment**: Hosted on Hugging Face Spaces  
- **Monitoring (optional)**: Visualisation of trends and model usage  

The application allows users to input customer attributes and receive churn predictions in an accessible format.

---

## Technology Stack
- **Python**  
- **Scikit-learn** (model development and evaluation)  
- **Streamlit** (interactive application)  
- **Git & GitHub** (version control and collaboration)  
- **Docker** (containerisation and deployment)  

---

## Running the API (Docker)
docker build -t reder-churn-api .

docker run -p 8000:8000 reder-churn-api

## Running the Streamlit App (Local)
streamlit run deploy/streamlit.py


## Notes
This repository focuses on the implementation of customer churn prediction using machine learning. The project demonstrates how predictive analytics can be applied to real-world telecommunications challenges to support retention and strategic decision-making.

