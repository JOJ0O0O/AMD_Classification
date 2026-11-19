# AMD_Classification
Project 3 Repository for the course FYS-STK3155/4155.

The code and report is created by:

- Johannes Schäfers
- Benedikt Schneider
- Ferdinand Nilius
- Benedikt Schneider

## Project Description

Age related macular degeneration is a disease mostly affecting people over the age of 50 and leads to a gradual decrease in vision.
The disease does not have a cure, but is however treatable and can progression can be slowed down significantly, if diagnosed early.
The diagnosis is often times done through images of the inside of one's eye, as there are relatively clear visual indicators for the disease.

We will attempt to use Machine Learning methods to tackle this task.
For this we will compare three different methods:

- Convolutional Neural Network
- Logistic Regression
- A tree based ensemble

## The Dataset

Our dataset can be found under the following link and is free to download:

Ali, Md Aiyub (2025), “AMDNet23: Fundus Image Dataset for Age-Related Macular Degeneration Disease Detection”, Mendeley Data, V1, doi: 10.17632/yj35kjgrv3.1
https://www.kaggle.com/datasets/orvile/macular-degeneration-disease-dataset/data

This dataset was first compiled and analyzed in the following paper:
Md. Aiyub Ali, Md. Shakhawat Hossain, Md. Kawsar Hossain, Subhadra Soumi Sikder, Sharun Akter Khushbu, Mirajul Islam,
AMDNet23: Hybrid CNN-LSTM deep learning approach with enhanced preprocessing for age-related macular degeneration (AMD) detection,
Intelligent Systems with Applications, Volume 21, 2024,

## Structure of the repository

TBD

## Running the code

All code used for our results is contained within this repository inside the folder "Code".
To install the required packages, run:
```bash
pip install -r requirements.txt
```

To execute all code (from scratch) execute:

**Windows:**
```powershell
python code\main.py
```

**Linux / macOS:**
```bash
python3 code/main.py
```

To only regenerate the figures execute:

**Windows:**
```powershell
python code\Figures.py
```

**Linux / macOS:**
```bash
python3 code/figures.py
```