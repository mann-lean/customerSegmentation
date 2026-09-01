# 🛍️ E-Commerce Customer Segmentation & RFM Engine

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-orange)
![UI](https://img.shields.io/badge/Frontend-Streamlit-red)

### 🚀 [Try the Live Dashboard Here](https://customersegmentationsolutions.streamlit.app/)

> **Stop guessing. Start targeting. An end-to-end machine learning pipeline that transforms raw retail transactions into a dynamic marketing playbook.**

---

## 👁️ Sneak Peek
__Segmentation Predictor__

![App Demo](artifacts\vis\clustClass.gif) 

__Business Analytics__

![App Demo](artifacts\vis\businesAnalytics.gif) 

__Customer Segentation__

![App Demo](artifacts\vis\cluster.png) 

__Customer Loopup__

![App Demo](artifacts\vis\customerLookup.gif) 

---

## 💡 The Vision: Why this exists
Retailers bleed money by treating all customers equally. Sending a 20% discount to a VIP who was going to buy anyway destroys profit margins, while ignoring a fading loyalist guarantees silent churn. 

This project is an intelligent segmentation engine. It analyzes historical buying habits—Recency, Frequency, and Monetary (RFM) metrics—and automatically assigns customers to actionable personas (e.g., *Enterprise Whales*, *At-Risk Customers*). It tells businesses exactly who to target, when to reach out, and what message to send to maximize Return on Investment (ROI).

---

## 📊 The Data 
**📂 [Link to the Original Dataset]([https://archive.ics.uci.edu/dataset/502/online+retail+ii])**

This project utilizes historical transactional data containing hundreds of thousands of real-world purchases. The raw ledger was heavily cleaned, filtered for anomalies (such as returned items and canceled orders), and grouped by unique Customer IDs to engineer the baseline RFM metrics used for modeling.

---

## 🔬 The Technical Edge: "Cluster-Then-Predict" Architecture
Most standard portfolios deploy brittle, distance-based unsupervised models (like K-Means) directly to production. Those are computationally heavy and easily broken by high-value extreme outliers. 

This project utilizes a highly scalable, **two-stage MLOps architecture**:

1. **Offline Discovery (Unsupervised + Heuristics):** Historical RFM data was log-transformed, standard-scaled, and clustered using K-Means. Mathematical centroids were then combined with strict, visually verified business-logic rules to safely isolate outliers and generate highly accurate ground-truth labels.
2. **Production Inference (Supervised RF):** A Random Forest Classifier (RFC) was trained to learn these complex, non-linear segment boundaries. 
3. **The Deployment Advantage:** Because tree-based models are natively immune to outliers, the live Streamlit dashboard **bypasses clunky feature scaling entirely**. It processes raw user inputs instantly to deliver lightning-fast, real-time segment predictions.

### 🔄 System Architecture
```text
[Raw Transactions] ➔ [RFM Feature Engineering] ➔ [K-Means + Visual Heuristics] ➔ [Labeled Dataset]
                                                                                        ↓
                     [Model Artifact: rfc.joblib] ⟵ [Random Forest Classifier] ⟵ [Labeled Training]
                                  ↓
[Streamlit UI] ➔ [Live Raw RFM Input] ➔ [RFC Inference Engine] ➔ [Targeted Marketing Strategy]
```

## 🖥️ Dashboard Features
* 📊 **Analytics:** Track overarching business KPIs (Total Revenue, Unique Customers, Average Basket Value).

* 🤖 **RFM Clustering:** Analyze offline cluster centroids and 3D behavioral distribution maps.

* 🔍 **Customer Lookup:** Search individual user profiles to extract raw transactional ledgers and micro-patterns.

* 🔮 **ML Classification:** A live inference simulator. Input raw RFM values to instantly predict a customer's persona using the pre-trained Random Forest engine.

* ⚙️ **Building Process:** A transparent look into the data engineering and algorithmic decisions behind the platform.

## 📂 Repository Structure
```text
├── app/app.py              # Main Streamlit application 
├── data/                   # Raw and processed datasets
├── artifacts/              # Serialized model artifacts (rfc.joblib)
├── research/               # Jupyter notebooks for Exploring data , EDA and model training
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🛠️ Tech Stack
* **Frontend:** Streamlit

* **Data Processing:** Pandas, NumPy

* **Machine Learning:** Scikit-Learn (K-Means, Random Forest)

* **Visualizations:** Plotly Express, Matplotlib, Seaborn

## 🚀 Quick Start (Run Locally)
__1. Clone the repository__

```bash 
git clone https://github.com/mann-lean/customerSegmentation.git
```
__2. Install dependencies__

```bash
pip install -r requirements.txt
```
__3. Launch the Engine__

```bash
streamlit run app.py
```
---
## 🚀 Future Development & Roadmap

- [ ] **Enterprise BI Integration:** Build an automated export pipeline to feed dynamic RFM segment data directly into **Power BI** for executive-level, cross-departmental reporting.
- [ ] **Backend API Decoupling:** Extract the Random Forest inference engine into a standalone **FastAPI** REST endpoint, allowing seamless integration with external CRM and email marketing platforms.
- [ ] **Automated MLOps Retraining:** Implement a data drift detection script that triggers an automated model retraining pipeline when customer purchasing habits shift.
- [ ] **GenAI Marketing Agent:** Integrate a local Large Language Model (LLM) to automatically draft personalized email copy based on the specific RFM segment a customer falls into.

## 📫 Let's Connect

<!-- **Mann**   -->
<!-- *Data Scientist | Machine Learning Engineer*   -->
Passionate about building end-to-end MLOps pipelines and data-driven business solutions.

* 💼 **LinkedIn:** [Click here to connect](http://www.linkedin.com/in/mann-32718a1b9)
* 🐙 **GitHub:** [Explore my other projects](https://github.com/mann-lean)
* ✉️ **Email:** [Reach out via email](mailto:mannk7062@gmail.com)
<!-- * 🌐 **Portfolio:** [View my complete portfolio](https://YOUR_PORTFOLIO_LINK) *(<- Optional: remove if you don't have a separate site)* -->