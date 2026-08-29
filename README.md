# multi-seller-fulfillment-analysis
Python code for the empirical analysis of multi-seller fulfillment and customer dissatisfaction in marketplace e-commerce.

# Multi-Seller Fulfillment and Customer Dissatisfaction

This repository contains the Python analysis used for the paper
**“Multi-Seller Fulfillment and Customer Dissatisfaction in Marketplace E-Commerce.”**

The study examines whether multi-seller order structure is associated with
customer dissatisfaction after accounting for delivery timing and other
observed order characteristics.

## Data

The analysis uses the **Brazilian E-Commerce Public Dataset by Olist**,
which is publicly available on Kaggle.

The dataset is downloaded automatically by `analysis.py` using `kagglehub`.
Raw data are not redistributed in this repository.

## Analysis

The script reproduces the main analytical workflow used in the paper,
including:

- order-level data preparation and validation;
- descriptive statistics;
- logistic regression Models A–L;
- alternative delivery-timing specifications;
- multi-item and product-diversity sensitivity analyses;
- seller-cluster bootstrap inference;
- average marginal effects;
- in-sample AUC comparison;
- diagnostic checks;
- figures and the consolidated results workbook.

All reported estimates are observational associations and should not be
interpreted as causal effects.

## Requirements

The final analysis was run with Python 3.13.15.

Install the required packages with:

```bash
pip install -r requirements.txt
