# 🚗 Old Car Price Prediction Using Machine Learning

## 📖 Project Overview
This project is a **Machine Learning-based Web Application** built with **Flask** that predicts the **resale price of used cars** based on features such as brand, year, fuel type, kilometers driven, and more.  

It demonstrates a complete **Data Science workflow** — from data cleaning and model training to web deployment and user interaction.  

---

## 🧠 Features
- 🧩 **Data Preprocessing** – Cleans and prepares raw car data.  
- 🤖 **Model Training** – Uses **Linear Regression** to predict car prices.  
- 🌐 **Flask Web App** – Simple and interactive interface for users to input car details and get predicted prices.  
- 📊 **Data Visualization Ready** – Cleaned dataset available for further EDA.  
- 💾 **Saved Model** – Trained model stored as `LinearRegressionModel.pkl` for fast prediction.  

---

## 🗂️ Project Structure
```
Old Car Price Prediction By Machine Lerning/
│
├── Application.py                 # Flask app entry point
├── Quack_Pridict.py               # Model training and logic
├── LinearRegressionModel.pkl      # Trained Linear Regression model
├── Cleaned_Car.csv                # Cleaned dataset
├── quikr_car.csv                  # Raw dataset
│
├── static/
│   └── css/
│       └── style.css              # Custom styling
│
└── templates/
    └── index.html                 # Web interface template
```

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/old-car-price-prediction.git
```

2. **Navigate to the project directory**
```bash
cd old-car-price-prediction
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

```bash
pip install flask scikit-learn pandas numpy
```

---

## 🚀 Run the App

Run the Flask application:
```bash
python Application.py
```

Then open your browser and visit:
```
http://127.0.0.1:5000/
```

---

## 💡 Usage

1. Enter car details like brand, year, fuel type, kilometers, etc.  
2. Click **Predict Price**.  
3. Instantly get the **estimated resale price** of your car.  

---

## 📸 Screenshots

<p align="center">
  <img src="images/Website%20overview_sc.png" alt="Web Interface" width="90%">
</p>

<p align="center">
  <img src="images/Website%20Overview-2.png" alt="Prediction Result" width="90%">
</p>


---


## 🧩 How It Works

1. **Data Collection:** Raw car listings were gathered from `quikr_car.csv`.  
2. **Data Cleaning:** Performed in `Quack_Pridict.py` to remove nulls and duplicates → output stored as `Cleaned_Car.csv`.  
3. **Model Training:** A **Linear Regression** model was trained on cleaned data.  
4. **Model Saving:** Model serialized using `pickle` as `LinearRegressionModel.pkl`.  
5. **Web Interface:** `Application.py` uses Flask to serve predictions dynamically.  

---

## 🛠️ Technologies Used
- 🐍 **Python 3**
- 🧮 **Scikit-learn**
- 📊 **Pandas / NumPy**
- 🌐 **Flask**
- 🎨 **HTML / CSS**

---



## 👨‍💻 Author
**Swaraj Darekar**  
📧 swarajdarekar3634@gmail.com  
💻 Passionate about Data Science & AI-based Web Apps  

---
✨ *Predict smarter, sell faster — using Machine Learning!* ✨
