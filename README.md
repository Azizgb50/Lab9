#  Lab 9 — Decision Trees & Random Forest

> **Predicting loan repayment using tree-based machine learning models on LendingClub data.**

---

##  Overview

This lab explores **Decision Tree** and **Random Forest** classifiers using real-world lending data from [LendingClub.com](https://www.lendingclub.com/). The goal is to predict whether a borrower will **fully repay their loan** based on their financial profile.

The dataset covers loans issued between **2007–2010**, before LendingClub went public.

---

##  Files

| File | Description |
|------|-------------|
| `02-Decision_Trees_and_Random_Forest_Project.ipynb` | Main project notebook |
| `loan_data.csv` | Cleaned LendingClub dataset (no NA values) |

---

## Dataset Features

| Column | Description |
|--------|-------------|
| `credit.policy` | 1 if customer meets LendingClub's underwriting criteria |
| `purpose` | Purpose of the loan (credit card, debt consolidation, etc.) |
| `int.rate` | Interest rate (e.g., 0.11 = 11%) |
| `installment` | Monthly installment amount |
| `log.annual.inc` | Natural log of borrower's annual income |
| `dti` | Debt-to-income ratio |
| `fico` | FICO credit score |
| `days.with.cr.line` | Number of days the borrower has had a credit line |
| `revol.bal` | Revolving balance (unpaid at end of billing cycle) |
| `revol.util` | Revolving line utilization rate |
| `inq.last.6mths` | Number of creditor inquiries in last 6 months |
| `delinq.2yrs` | Times 30+ days past due in past 2 years |
| `pub.rec` | Number of derogatory public records |
| `not.fully.paid` | **Target** — 1 if loan was not fully repaid |

---

## Workflow

### 1. Exploratory Data Analysis
- FICO score distributions by `credit.policy` and `not.fully.paid`
- Loan count by purpose (countplot)
- FICO vs interest rate (jointplot + lmplot)

### 2. Data Preparation
- Convert categorical `purpose` column to dummy variables using `pd.get_dummies`
- Train/Test split (70% train, 30% test)

### 3. Modeling
- **Decision Tree Classifier** — baseline model
- **Random Forest Classifier** — ensemble model (600 trees)

### 4. Evaluation
- Classification report (precision, recall, F1-score)
- Confusion matrix

---

##  Results

| Model | Accuracy | Precision (Default) | Recall (Default) |
|-------|----------|---------------------|-----------------|
| Decision Tree | ~73% | 0.16 | 0.20 |
| Random Forest | ~85% | 0.56 | 0.01 |

###  Key Finding
The **Random Forest** achieved higher overall accuracy but had nearly **zero recall on defaulters** (class 1), meaning it almost always predicted loans as "fully paid." This is a classic **class imbalance** problem — only ~16% of loans in the dataset were not fully paid.

The **Decision Tree** was more balanced and better at actually catching risky borrowers, which is more valuable from an investor's perspective.

---

##  Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
```

---

##  How to Run

1. Clone the repository
2. Place `loan_data.csv` in the same folder as the notebook
3. Open `02-Decision_Trees_and_Random_Forest_Project.ipynb` in Jupyter
4. Run all cells top to bottom

---

##  Course

**Machine Learning Course — Lab 9**  
Topic: Tree-Based Methods (Decision Trees & Random Forests)
