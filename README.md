# RetailIQ – Customer Segmentation for a Retail Chain

## About the Project
RetailIQ is a machine learning-based customer segmentation system that groups retail customers based on their purchasing behaviour.

It uses RFM analysis and clustering algorithms to identify customer segments and suggest suitable marketing strategies.

## Problem Statement
Retail businesses have customers with different purchasing patterns. Identifying high-value customers and understanding customer behaviour can help businesses improve customer retention and marketing strategies.

## Objectives
- Analyze customer purchasing behaviour.
- Perform RFM analysis.
- Apply K-Means and Hierarchical Clustering.
- Compare clustering performance.
- Visualize customer segments using PCA.
- Generate customer personas and marketing recommendations.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- SciPy

## Dataset
UCI Online Retail Dataset.

Dataset Link: https://archive.ics.uci.edu/dataset/352/online+retail

Download the dataset separately and upload it through the application.

## Methodology
1. Data Cleaning
2. RFM Feature Engineering
3. Log Transformation and Standardization
4. Elbow Method and Silhouette Analysis
5. K-Means Clustering
6. Hierarchical Clustering
7. PCA Visualization
8. Customer Persona Generation

## Results

| Model | Silhouette Score |
|---|---:|
| K-Means | 0.4328 |
| Hierarchical Clustering | 0.4040 |

The selected K-Means model generated two customer segments.

- Cluster 0: Low-Value / Occasional Customers
- Cluster 1: Loyal / High-Value Customers

The first two PCA components explain 93.87% of the variance.

## Application Features
- Upload retail transaction data.
- Perform customer segmentation.
- View clustering analysis.
- Compare machine learning models.
- Visualize customer groups.
- Explore customer personas.
- Download segmentation results.

## How to Run

Install the required libraries:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn scipy openpyxl
```

Run the application:

```bash
python -m streamlit run app.py
```

## Future Scope
- Real-time retail data integration.
- Advanced clustering techniques.
- Customer lifetime value prediction.
- Personalized marketing recommendations.

## Author

**Gauri Dyappa**
MSc Artificial Intelligence and Machine Learning Project

**Project:** RetailIQ – Customer Segmentation for a Retail Chain
