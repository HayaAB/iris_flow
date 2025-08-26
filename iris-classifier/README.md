# 🌸 Iris Classification App

A simple **Machine Learning project** that demonstrates the process of data analysis, model training, and building an interactive web application with **Streamlit**.  

This project uses the classic **Iris dataset** to classify flower species based on petal and sepal measurements.  

---

## 👩‍💻 Author
**Haya Albhaisi**

---

## 📊 Project Workflow
1. **Exploratory Data Analysis (EDA)**  
   - Conducted in Jupyter Notebook.  
   - Explored the dataset with descriptive statistics and visualizations.  
   - Checked class distribution and feature relationships.  

2. **Model Training**  
   - Preprocessing with **StandardScaler**.  
   - Model: **RandomForestClassifier** (scikit-learn).  
   - Combined into a **Pipeline** for reproducibility.  
   - Achieved high accuracy on the test set.  

3. **Model Persistence**  
   - Saved the trained pipeline using **joblib** into the `models/` directory.  

4. **Interactive App**  
   - Built with **Streamlit** in `app.py`.  
   - Users can input flower measurements with sliders.  
   - The app predicts the species and shows class probabilities with a bar chart.  

---

## 📂 Repository Structure
```
iris_flow/
│
├── notebooks/                # Jupyter notebooks for EDA and training
│   └── 01_eda_training.ipynb
│
├── data/
│   └── processed/            # Processed dataset (optional)
│
├── models/
│   └── iris_pipeline.joblib  # Trained model
│
├── app.py                    # Streamlit web app
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/iris_flow.git
   cd iris_flow
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser at:
   ```
   http://localhost:8501
   ```

---


## 🌐 Deployment
This project can be deployed easily on **[Streamlit Community Cloud](https://streamlit.io/cloud)** by connecting the GitHub repository.

---

## 📌 Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Joblib  
- Streamlit  

---

## ✨ Acknowledgements
- Dataset: [UCI Machine Learning Repository - Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)  
- Streamlit Documentation: [https://docs.streamlit.io](https://docs.streamlit.io)
