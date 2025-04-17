# 🧠 MATLAB AI Project – Heart Disease Prediction

This project implements a heart disease prediction model using MATLAB. It includes statistical data analysis, classification modeling, and hypothesis testing to identify key predictors of cardiovascular risk. The model was trained using public clinical datasets and validated with k-fold cross-validation for performance evaluation.

## 📂 Project Structure
matlab-heart-ai-project/
├── code/
│   └── Heart_Data_Analysis.m               # Main analysis + AI model training script
├── models/
│   ├── HeartDiseaseModel.mat               # Original trained model
│   └── UpdatedHeartDiseaseModel.mat        # Refined or re-trained model
├── data/
│   ├── heart.csv                           # Primary dataset (clinical data)
│   ├── heart.xlsx                          # Alternative data format
│   └── cardio_train.csv                    # Additional dataset for testing/generalization
├── notes/
│   └── Hypothesis_Conclusion.ps            # Statistical inference and conclusion summary


## ⚙️ What It Does

- 📊 Loads and processes real-world cardiovascular datasets  
- 🧠 Trains classification models using MATLAB’s ML toolbox  
- 🔁 Applies k-fold cross-validation for training/testing separation  
- ✅ Evaluates model accuracy on multiple datasets  
- 📈 Includes statistical hypothesis testing on feature significance

## 🔬 Technologies Used

- MATLAB (R2023 or newer recommended)  
- MATLAB Machine Learning Toolbox  
- .mat model saving/loading  
- .csv and .xlsx data imports

## 💡 Example Use Case

> Load `heart.csv`, train a model, and predict heart disease presence based on features like cholesterol, age, and chest pain type. Evaluate accuracy on an alternate dataset and visualize decision metrics.

## 🧠 Author

**Samuel de Souza**  
Biomedical Engineering @ Drexel University  
[GitHub](https://github.com/SamAugusto)  
[LinkedIn](https://www.linkedin.com/in/samuel-de-souza-0b1302226)

---

##  Tags

`#MATLAB` `#MachineLearning` `#HeartDisease` `#BiomedicalEngineering` `#AI`
