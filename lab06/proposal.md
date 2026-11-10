# Simple Project Plan: Fintech Fraud Detection

## 1. Project Topic
**Detecting Fraudulent Transactions and Anomalies in Fintech**

Protecting peer-to-peer (P2P) transfers and instant payments (directly relevant to the Polish market, such as the BLIK system and phone-number transfers).

---

## 2. Dataset to Use
The **PaySim** dataset (publicly available on Kaggle, based on real mobile money transaction logs).

## 3. What Machine Learning Does in This Project
1. **Learns Behavioral Patterns:** The model studies thousands of normal transfers and rare fraud cases to spot suspicious combinations of features.
2. **Calculates a Risk Score:** For each incoming payment, the model outputs a probability from 0% to 100% indicating fraud likelihood.
3. **Applies a Decision Threshold:** If the risk score exceeds a chosen cutoff (e.g., 70%), the transaction is held for manual review or blocked.
4. **Explains the Reason (SHAP):** It shows the top 2–3 drivers behind the alert (e.g., *"100% of balance drained"* + *"transaction initiated at 4 AM"*).

---

## 5. Step-by-Step Implementation Plan
1. **Step 1: Data Setup:** Download PaySim and inspect the first 50–100 rows in a Jupyter Notebook.
2. **Step 2: Exploration (EDA):** Check the fraud percentage and build 2–3 key charts showing amount distributions and transaction types.
3. **Step 3: Feature Creation:** Add 2–4 new columns (balance differences, zero-balance flags, transaction frequency).
4. **Step 4: Train the Model:** Run a fast baseline model (`LightGBM` or `Random Forest`) and evaluate how many real frauds it catches.
5. **Step 5: Resume Summary:** Calculate the bottom-line business value — estimated money saved versus number of false alarms.