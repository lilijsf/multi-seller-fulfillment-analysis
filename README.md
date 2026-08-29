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

Dataset: [Brazilian E-Commerce Public Dataset by Olist] (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

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
```

## Running the analysis

Run:
```bash
analysis.py
```

The script downloads the public Olist dataset and saves the generated
tables, workbook, and figures in the outputs directory.

The seller-cluster bootstrap uses 1,000 replications, so the full analysis
may take some time to complete.

## Output

The script generates:
- python_results.xlsx
- descriptive and robustness outputs
- fulfillment and delivery figures
- adjusted probability and coefficient figures

These files are generated automatically and are not stored in the repository.

## Citation

If you use this code, please cite the associated paper:

**Multi-Seller Fulfillment and Customer Dissatisfaction in Marketplace E-Commerce**

Full publication details will be added after publication.

## License

This repository is released under the MIT License.
