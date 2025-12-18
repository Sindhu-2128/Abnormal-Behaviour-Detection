# Abnormal-Behaviour-Detection
Abnormal Behaviour Detection project uses optical flow and machine learning to identify unusual crowd movement in video surveillance. Motion features are extracted from video frames and classified using SVM, Logistic Regression, and KNN to distinguish normal and abnormal behavior.
## Overview
This project focuses on detecting abnormal human behavior in crowded environments using computer vision and machine learning techniques. By analyzing motion patterns through optical flow, the system differentiates normal and abnormal crowd behavior in video sequences.
The approach is privacy-preserving, as it relies on motion information rather than facial or personal identity data.

## Objectives

Detect abnormal behavior in crowded public environments

Extract motion-based features using optical flow

Classify crowd behavior as normal or abnormal

Support intelligent video surveillance systems

## Methodology

- Video Preprocessing

- Convert video frames to grayscale

- Apply normalization and morphological filters

- Feature Extraction

- Optical Flow (U, V motion vectors)

- Machine Learning Models

- Support Vector Machine (SVM)

- Logistic Regression

- K-Nearest Neighbors (KNN)

- Performance comparison across classifiers

## Technologies Used

Language: Python 3.12

Libraries & Tools:

OpenCV

NumPy

Scikit-learn

Scikit-image

Matplotlib

Environment: Anaconda, Jupyter Notebook


## Dataset

UCF Crime dataset

229 video frames for abnormal Behaviour

341 pedestrian instances video frame for normal Behaviour

Used for validation of segmentation and classification

## Key Results & Insights

Optical flow effectively captures motion irregularities in crowds

SVM achieved better classification performance compared to KNN

The system successfully distinguishes abnormal behavior patterns

Suitable for real-time surveillance and public safety monitoring

