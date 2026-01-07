# AI-Powered Image Similarity Search and Recommendation System

## 📌 Overview
This project implements an AI-powered image similarity search and recommendation system using Deep Learning. A pre-trained MobileNetV2 model is used to extract visual features from images, and cosine similarity is applied to recommend visually similar images.

## 🚀 Key Features
- Deep Learning based feature extraction
- Image similarity using cosine similarity
- Lightweight and efficient CNN (MobileNetV2)
- Custom dataset support

## 🧠 Technologies Used
- Python
- TensorFlow & Keras
- MobileNetV2
- NumPy
- OpenCV
- scikit-learn

## ⚙️ How It Works
1. Images are loaded from the dataset folder  
2. MobileNetV2 extracts deep feature vectors  
3. Feature vectors are stored locally  
4. Cosine similarity is calculated  
5. Top similar images are recommended  

## ▶️ How to Run

### 1. Install required libraries:
bash
pip install tensorflow opencv-python scikit-learn numpy


### 2. Run feature extraction:
bash
python feature_extraction.py


### 3. Run similarity search:
bash
python similarity_search.py

📊 Output

The system displays the query image name along with a list of visually similar images and their similarity scores.

📌 Applications
	•	E-commerce product recommendation
	•	Fashion image search
	•	Visual search systems

  🔮 Future Scope
	•	Web-based interface
	•	Large-scale image databases
	•	Real-time image upload

  👨‍🎓 Author

Shobhraj Bhattacharjee
University- The Assam Kaziranga University 
B.Tech – Computer Science Engineering
Enrolment id - ET24BTHCS095

📜 License

This project is a private academic project. All rights are reserved.

---

# ✅ 2️⃣ requirements.txt  

```txt
tensorflow
opencv-python==4.8.1.78
numpy==1.26.4
scikit-learn
