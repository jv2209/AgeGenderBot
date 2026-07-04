![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-yellow)
# 👤 AgeGenderBot

An AI-powered application that detects a person's **age** and **gender** in real time using **Python**, **OpenCV**, and pre-trained **Caffe Deep Learning** models.

## 🚀 Features

- 🎥 Real-time webcam face detection
- 👤 Age prediction
- 🚻 Gender prediction
- ⚡ Fast inference using OpenCV DNN

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Caffe Deep Learning Models
## 📥 Required Models

Download the following pre-trained Caffe models and place them inside the `models/` folder:

- age_net.caffemodel
- gender_net.caffemodel
- res10_300x300_ssd_iter_140000.caffemodel
## 🎥 Demo

Real-time age and gender detection using a webcam powered by OpenCV and Deep Learning.

## 📷 Screenshot

![AgeGenderBot](screenshots/age-gender-detector.jpg)

## 📦 Installation

```bash
pip install -r requirements.txt
```

## ▶️ Run

```bash
python main.py
```
## 📊 Output

The application:

✔ Detects faces

✔ Predicts age

✔ Predicts gender

✔ Draws a bounding box around the face

✔ Displays prediction confidence

## 📁 Project Structure

```
AgeGenderBot/
├── main.py
├── models/
├── screenshot/
├── README.md
├── requirements.txt
└── .gitignore
```

## 🚀 Future Improvements

- Support multiple face detection
- Improve age prediction accuracy
- Add GUI interface
- Export detection results
- Deploy as a web application
