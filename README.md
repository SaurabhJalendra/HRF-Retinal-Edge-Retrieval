# HRF Retinal Image Content-Based Retrieval System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![scikit-image](https://img.shields.io/badge/scikit--image-0.25%2B-orange.svg)](https://scikit-image.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Project Overview

This project implements a **Content-Based Image Retrieval (CBIR)** system specifically designed for retinal fundus images using the **High-Resolution Fundus (HRF)** dataset. The system employs advanced computer vision techniques and handcrafted feature engineering to enable efficient retrieval of similar retinal images based on visual content.

### Key Objectives
- Develop a robust CBIR system for medical image analysis
- Extract and combine multiple handcrafted features for comprehensive image representation
- Compare different feature engineering techniques for retinal image classification
- Provide insights into feature importance and model performance

## Dataset

**HRF (High-Resolution Fundus) Dataset**
- **Total Images**: 45 high-resolution retinal fundus images
- **Categories**: 
  - **Healthy**: 15 images
  - **Diabetic Retinopathy**: 15 images  
  - **Glaucoma**: 15 images
- **Resolution**: High-quality fundus photographs
- **Format**: JPEG/JPG images

## Features Implemented

### 1. **Handcrafted Feature Extraction**
- **HOG (Histogram of Oriented Gradients)**: Captures edge and gradient information
- **LBP (Local Binary Patterns)**: Analyzes local texture patterns
- **Edge Detection**: Sobel, Canny, and Laplacian edge features
- **Gabor Filters**: Texture analysis using frequency domain features
- **GLCM (Gray-Level Co-occurrence Matrix)**: Statistical texture descriptors
- **Color Histograms**: RGB and HSV color distribution features

### 2. **Feature Combination Strategies**
- Individual feature performance analysis
- Feature concatenation and fusion techniques
- Dimensionality reduction using PCA
- Feature selection using statistical methods

### 3. **Machine Learning Models**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**

### 4. **Evaluation Metrics**
- Classification accuracy
- Precision, Recall, F1-score
- Confusion matrices
- Feature importance visualization
- Cross-validation performance

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/HRF-Retinal-Edge-Retrieval.git
cd HRF-Retinal-Edge-Retrieval
```

2. **Create virtual environment**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Libraries
The project dependencies are specified in `requirements.txt`:
```
opencv-python>=4.8.0
scikit-image>=0.25.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
tqdm>=4.65.0
pillow>=10.0.0
scipy>=1.10.0
joblib>=1.3.0
ipykernel>=6.25.0
notebook>=7.0.0
```

## Usage

### Running the Jupyter Notebook
```bash
jupyter notebook HRF-Retinal-Edge-Retrieval.ipynb
```

### Key Notebook Sections
1. **Data Loading and Exploration**
2. **Feature Extraction Pipeline**
3. **Model Training and Evaluation**
4. **Performance Comparison**
5. **Feature Importance Analysis**
6. **Results Visualization**

## Results

*Note: Results will be updated after running the complete analysis.*

### Model Performance Comparison
The notebook provides comprehensive analysis of different feature combinations and their performance on the HRF dataset.

### Key Insights
- Feature combination strategies and their effectiveness
- Comparative analysis of different machine learning models
- Feature importance rankings for retinal image classification
- Performance metrics across different classification scenarios

## Technical Highlights

### Feature Engineering Pipeline
The project implements a comprehensive feature extraction pipeline combining multiple computer vision techniques for robust retinal image analysis.

### Model Evaluation Framework
- **Stratified K-Fold Cross-Validation**
- **Statistical significance testing**
- **Feature importance ranking**
- **Comprehensive visualization**

## Project Structure
```
HRF-Retinal-Edge-Retrieval/
├── HRF/                                          # Dataset directory
│   ├── healthy/                                  # Healthy retinal images (15 images)
│   ├── diabetic_retinopathy/                     # DR images (15 images)
│   └── glaucoma/                                 # Glaucoma images (15 images)
├── venv/                                         # Virtual environment
├── Connection Sensitive Attention U-NET...pdf   # Reference paper
├── HRF-Retinal-Edge-Retrieval.ipynb            # Main notebook
├── README.md                                     # Project documentation
└── requirements.txt                              # Dependencies
```

## Educational Value

This project demonstrates:
- **Handcrafted feature engineering** in computer vision
- **Medical image analysis** techniques
- **Content-based image retrieval** systems
- **Feature fusion** and selection strategies
- **Performance evaluation** methodologies

## Additional Resources

- **Reference Paper**: Connection Sensitive Attention U-NET for Accurate Retinal Vessel Segmentation (included in repository)
- **Dataset**: HRF (High-Resolution Fundus) database for vessel segmentation studies

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- **HRF Dataset**: Thanks to the creators of the High-Resolution Fundus dataset
- **OpenCV Community**: For excellent computer vision tools
- **scikit-image**: For comprehensive image processing capabilities
- **scikit-learn**: For machine learning implementations

## Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

**Star this repository if you found it helpful!**
