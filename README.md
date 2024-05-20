# Brain_Tumor_Detection_EfficientNet
# Brain Tumor Detection 🧠

Welcome to the Brain Tumor Detection project This web application uses deep learning to identify brain tumors from medical images, providing you with accurate diagnoses to assist in effective treatment. Let's dive in!

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation & Metrics](#evaluation--metrics)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Contact](#contact)

## Project Overview
Brain tumor detection is a complex challenge, but our project tackles it head-on We're using a convolutional neural network (CNN) to classify brain tumors into four categories: Glioma, Meningioma, Pituitary, and No Tumor. The web application integrates the trained CNN model to make predictions on user-uploaded images.

## Dataset
The dataset includes 7023 medical images across four classes:

- **Glioma Tumor**: 1621 images
- **Meningioma Tumor**: 1645 images
- **No Tumor**: 2000 images
- **Pituitary Tumor**: 1757 images

Organized as follows:

## Installation
1. **Clone the Repository**:  
    ```bash
    git clone https://github.com/your-username/brain-tumor-detection.git
    ```

2. **Install Dependencies**:  
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare the Dataset**:  
    Ensure you have the dataset in the specified directory structure (`/dataset/Training` and `/dataset/Testing`).

## Getting Started
1. **Run the Flask Application**:  
    ```bash
    python Webapp.py
    ```

2. **Navigate to the Web App**:  
    Open your web browser and go to `http://127.0.0.1:5000/`.

3. **Upload Your Image**:  
    Upload an image of a brain scan through the web interface and receive predictions and confidence levels.

## Model Architecture
Our updated CNN architecture leverages the latest advancements in deep learning for enhanced performance. Key components include:

- **EfficientNetB3 Base Model**: The pre-trained EfficientNetB3 model is used as the base model, excluding the top classification layer. Utilized for robust feature extraction, benefiting from its proven efficiency and effectiveness in image recognition tasks.
- **Batch Normalization**:  Helps in stabilizing the training process and reducing the impact of internal covariate shift.
- **Dense Layers**: Fully connected layers with L1 and L2 regularization to prevent overfitting.
- **Dropout Layer**: Introduces dropout regularization to further reduce overfitting.
- **Softmax Output Layer**: Configured for multi-class classification, accurately predicting among the four tumor categories.

## Training the Model
The training process has been refined to maximize the model's accuracy and efficiency:

1. **Advanced Data Augmentation**: Employed more aggressive data augmentation techniques to increase the diversity of the training set, enhancing the model's ability to generalize.
2. **Optimizer Selection**: Experimented with various optimizers, settling on a combination that offers a balance between convergence speed and stability.
3. **Regularization Strategies**: Applied stricter regularization techniques to mitigate overfitting, ensuring the model performs well on unseen data.
4. **Early Stopping Optimization**: Implemented a more sophisticated early stopping mechanism to halt training at the optimal point, based on validation loss trends.

## Evaluation & Metrics
The model's performance is rigorously assessed using a comprehensive suite of evaluation metrics:

- **Enhanced Precision and Recall**: Calculated for each tumor category to ensure the model accurately identifies positive cases without too many false positives or negatives.
- **F1-Score**: Used as a harmonic mean of precision and recall, providing a single metric that balances both types of errors.
- **Confusion Matrix Analysis**: Conducted a detailed analysis of the confusion matrix to gain insights into the model's classification errors, guiding further improvements.

## Deployment
Deploy the trained model as a web application using Flask. Options for deployment include local servers or cloud platforms (AWS, GCP). Consider using Docker for containerization.

## Contributing
We welcome contributions If you encounter issues or have suggestions, please open an issue or submit a pull request.

## Contact
If you have questions or feedback, reach out to us:

- **Email**: aryansin2468@gmail.com
- **LinkedIn**: [Aryan Singh](https://www.linkedin.com/in/aryan-singh-162560260/)

Thank you for your interest in the project Let's collaborate and make an impact in the medical domain!
