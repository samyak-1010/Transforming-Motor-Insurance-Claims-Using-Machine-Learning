# Transforming Motor Insurance Claims Using Machine Learning

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Success-brightgreen.svg)

## 🚀 Overview
This project leverages **Machine Learning and NLP techniques** to improve motor insurance claim verification, detect fraudulent claims, and enhance decision-making through automated **text analysis, anomaly detection, and recommendation engines**.

## 📌 Key Features
- **Part Mapping Techniques** 🔍  
  - **Fuzzy Matching** for textual similarity detection
  - **TF-IDF with Cosine Similarity** for semantic understanding
- **Association Rule Mining** 📊  
  - **Apriori Algorithm** to uncover patterns in claims
- **Collaborative Filtering** 🤝  
  - **Dimensionality Reduction (SVD)** for accurate recommendations
- **Fraud Detection** ⚠️  
  - **Isolation Forest** to detect anomalies in claim amounts
- **End-to-End Automation** ⚡  
  - Seamless data pipeline from preprocessing to visualization

---

## 📂 Dataset
The project utilizes **survey and garage claim data**, focusing on part descriptions and claim details.

**Example Columns:**
- `REFERENCE_NUM`: Unique claim reference number
- `VEHICLE_MODEL_CODE`: Vehicle model identifier
- `CLAIMNO`: Claim number
- `PART_DESCRIPTION`: Descriptions of damaged parts
- `CLAIM_AMOUNT`: Amount claimed for repairs

> **Note**: Sample datasets can be found in `data/` directory.

---

## 🔧 Installation
```sh
# Clone the repository
git clone https://github.com/your-repo/motor-insurance-ml.git
cd motor-insurance-ml

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Usage
```sh
# Run the main pipeline
python main.py
```

### 🏁 Step-by-Step Execution
1. **Preprocessing & Cleaning**: Loads, merges, and cleans datasets
2. **Part Matching**: Uses **Fuzzy Matching & TF-IDF Cosine Similarity**
3. **Pattern Recognition**: Applies **Apriori Algorithm** for frequent itemsets
4. **Recommendation Engine**: Uses **Collaborative Filtering with SVD**
5. **Fraud Detection**: Detects anomalies using **Isolation Forest**
6. **Visualization**: Generates plots and insights

---

## 📊 Results & Visualizations
### Fraudulent Claim Detection Example
```python
# Detect anomalies
from sklearn.ensemble import IsolationForest
model = IsolationForest(contamination=0.05)
fraud_labels = model.fit_predict(claim_data[['CLAIM_AMOUNT']])
```

📌 **Sample Visualization:**

![Fraud Detection](https://user-images.githubusercontent.com/example/fraud-detection.png)

---

## 🎯 Business Prospects
✔️ **Faster and more accurate claim verification**  
✔️ **Reduced fraud detection time, lowering financial losses**  
✔️ **Automated and scalable solution for insurance firms**

---

## 🏆 Conclusion
Our solution **bridges the gap between manual claim validation and automated decision-making**, reducing fraud, improving efficiency, and ensuring data-driven insights. The **hybrid approach of NLP, ML, and data mining** makes this model a valuable asset in transforming the motor insurance industry. 🚀

---

## 🤝 Contributing
We welcome contributions! Please check the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📝 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
