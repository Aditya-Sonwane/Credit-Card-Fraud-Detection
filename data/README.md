# Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle, provided by the ULB Machine Learning Group.

- 284,807 transactions made by European cardholders in September 2013
- 492 fraud cases (0.172% of all transactions)
- Features `V1`–`V28` are PCA-transformed for confidentiality; `Time` and `Amount` are the only original features

## How to use

1. Download `creditcard.csv` from the Kaggle link above (requires a free Kaggle account)
2. Place the file in this `data/` folder
3. The file is excluded from version control via `.gitignore` due to its size (~150 MB) and Kaggle's licensing terms