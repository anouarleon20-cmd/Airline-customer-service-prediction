# ✈️ Airline Customer Satisfaction Prediction & Analysis

A comprehensive machine learning pipeline for predicting airline passenger satisfaction using multi-source customer data, combining sentiment analysis, behavioral patterns, and explainable AI techniques.

## 📋 Overview

This project analyzes the **Airline Customer Holiday Booking Dataset** to understand customer behavior, identify satisfaction drivers, and build predictive models for passenger satisfaction. By integrating structured booking data, customer reviews, and in-flight survey responses, we deliver actionable insights for improving airline services in a competitive market.

## 🎯 Objectives

- **Data Integration & Cleaning**: Process and merge multiple data sources including passenger reviews, booking records, customer comments, and satisfaction surveys
- **Sentiment Analysis**: Extract sentiment scores from customer review content using VADER sentiment analysis
- **Exploratory Data Analysis**: Identify popular flight routes, booking patterns, and traveler preferences across different segments
- **Predictive Modeling**: Develop binary classification models (statistical ML / shallow FFNN) to predict passenger satisfaction (Rating ≥ 5 vs < 5)
- **Model Explainability**: Apply XAI techniques (SHAP & LIME) to interpret predictions and identify key satisfaction drivers

## 📊 Datasets

The project utilizes four primary data sources:

| Dataset | Description |
|---------|-------------|
| `AirlineScrappedReview.csv` | Customer reviews with ratings, routes, travel dates, and traveler types |
| `Customer_comment.csv` | Passenger feedback with flight operations and service metrics |
| `Passanger_booking_data.csv` | Booking behavior and ancillary service preferences |
| `Survey_data_Inflight_Satisfaction.csv` | In-flight experience survey responses |

## 🔍 Key Analysis Questions

**Data Engineering Insights:**
- What are the top 10 most popular flight routes?
- What is the distribution of bookings across flight hours?
- Which traveler type and class combination yields the highest and lowest ratings?

**Predictive Task:**
- Binary classification: Satisfied (Rating ≥ 5) vs Dissatisfied (Rating < 5)

## 🛠️ Technologies & Tools

- **Languages**: Python 3.x
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, TensorFlow/PyTorch
- **Sentiment Analysis**: VADER (NLTK)
- **Explainable AI**: SHAP, LIME
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Jupyter Notebook

## 📁 Repository Structure

```
.
├── notebooks/
│   └── airline_satisfaction_analysis.ipynb    # Main analysis notebook
├── data/
│   ├── raw/                                    # Original datasets
│   └── processed/                              # Cleaned and engineered datasets
├── models/
│   └── trained_models/                         # Saved model artifacts
├── src/
│   ├── preprocessing.py                        # Data cleaning functions
│   ├── feature_engineering.py                  # Feature creation
│   ├── model_training.py                       # Model development
│   └── inference.py                            # Prediction function
├── reports/
│   ├── figures/                                # Visualizations
│   └── analysis_report.pdf                     # Detailed findings
├── requirements.txt                            # Project dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook
```

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/airline-satisfaction-prediction.git
cd airline-satisfaction-prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download VADER lexicon
```python
import nltk
nltk.download('vader_lexicon')
```

### Usage

1. **Run the analysis notebook**
```bash
jupyter notebook notebooks/airline_satisfaction_analysis.ipynb
```

2. **Use the inference function**
```python
from src.inference import predict_satisfaction

# Example prediction
result = predict_satisfaction(
    traveller_type="Solo Leisure",
    class_type="Economy Class",
    rating=4,
    verified=True,
    sentiment_score=0.6
)
print(f"Prediction: {result}")
```

## 📈 Model Performance

Models are evaluated using:
- **Accuracy**: Overall prediction correctness
- **Precision**: Correct positive predictions ratio
- **Recall**: True positive detection rate
- **F1-Score**: Harmonic mean of precision and recall

Detailed performance metrics and comparison between models are available in the analysis notebook.

## 🔬 Explainability Outputs

The project includes comprehensive XAI analysis:

- **SHAP (SHapley Additive exPlanations)**
  - Feature importance rankings
  - Summary plots showing global feature impact
  - Force plots for individual predictions

- **LIME (Local Interpretable Model-agnostic Explanations)**
  - Instance-level prediction explanations
  - Feature contribution visualization

## 📄 Deliverables

✅ Reproducible Jupyter Notebook with complete analysis workflow  
✅ Cleaned datasets with sentiment analysis column  
✅ Analytical report with EDA insights and visualizations  
✅ Trained predictive models (ML & Neural Network)  
✅ XAI outputs (SHAP plots, LIME explanations)  
✅ Production-ready inference function  

## 🎓 Academic Context

**Course**: CSEN 903 - Machine Learning  
**Institution**: German University in Cairo  
**Instructor**: Dr. Nourhan Ehab  
**Milestone**: 1 - Airline Customer Holiday Booking Analysis

## 📝 License

This project is part of academic coursework at GUC.

## 👥 Contributors

Ahmed Alaa

Anouar Leon

ALi Waleed

Mohamed Hefny

## 🙏 Acknowledgments

- Dr. Nourhan Ehab for project guidance
- Airline Customer Holiday Booking Dataset providers
- German University in Cairo, Faculty of Media Engineering and Technology

---

**Note**: This repository was developed as part of the CSEN 903 course requirements. Please ensure academic integrity guidelines are followed when referencing this work.
