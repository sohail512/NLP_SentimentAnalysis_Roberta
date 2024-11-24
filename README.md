# Sentiment Analysis with RoBERTa Model

This project demonstrates the use of **RoBERTa**, a state-of-the-art transformer model, for sentiment analysis. The goal is to classify textual phrases into sentiment categories using a fine-tuned version of RoBERTa on a labeled dataset.

---

## 🚀 **Project Overview**

- **Objective:** Build a deep learning model to classify phrases into sentiment categories.
- **Model:** Fine-tuned [RoBERTa](https://huggingface.co/roberta-base) (a robustly optimized BERT model) using the PyTorch framework.
- **Application in Sentiment Analysis:** Explore how we applied RoBERTa to sentiment analysis, specifically on a dataset of movie reviews.
- **Environment Setup:** Learn about the libraries and tools we used, including PyTorch, Transformers, and the Hugging Face library.
- **Data Preparation:** Details on loading and preparing the training data from a provided file.
- **Defining Variables:** Key variables defined for training and validation.
- **Dataset and DataLoader:** Creation of a custom dataset class and corresponding data loaders.
- **Neural Network Architecture:** Introduction to the fine-tuning neural network based on RoBERTa.
- **Loss Function and Optimizer:** Information on the loss function and optimizer used during training.
- **Training Function:** Explanation of the training process and how the model learns from the data.
- **Validation Function:** Insights into the validation stage to evaluate the model's performance on unseen data.
-  **Outcome:** Achieved competitive accuracy and implemented a scalable training and validation pipeline.
-  **Web App:** A **Streamlit bot** that allows users to input reviews and get sentiment analysis results in real-time.

  ### Prerequisites

- Ensure you have Python installed (preferably Python 3.x).
- Download the training data file `train.tsv.zip` from [Kaggle](https://www.kaggle.com/c/movie-review-sentiment-analysis-kernels-only/data) and place it in the project directory.


---
## 🖥️ **Streamlit Bot**

We developed an interactive **Streamlit** web app where users can:

1. Enter a review or a phrase in the input box.
2. Receive the sentiment classification (e.g.,Somewhat Positive, Positive,Somewhat Negative, Negative, Neutral,) instantly.
   ---

## 🛠️ **Features**

1. **Tokenization and Preprocessing:** 
   - Utilized the `RobertaTokenizer` to tokenize input text.
   - Handled padding, truncation, and special tokens effectively.

2. **Custom Dataset Class:** 
   - Created a `SentimentData` class to preprocess and structure the data for the PyTorch DataLoader.

3. **Model Architecture:**
   - Pre-trained RoBERTa model as the backbone.
   - Fully connected layers for sentiment classification.
   - Dropout layers to reduce overfitting.

4. **Training and Validation Pipelines:**
   - Batch training with performance monitoring every 5000 steps.
   - Evaluation on unseen data to measure generalization.

5. **Performance Metrics:**
   - Used accuracy and loss as evaluation metrics.
   - Real-time logging for training and validation steps.

---

## ⭐ **Acknowledgements**

We would like to acknowledge the following for their invaluable contributions and resources that made this project possible:

1. **[Hugging Face](https://huggingface.co):**  
   - For providing pre-trained models like RoBERTa and powerful tools for NLP development.

2. **[PyTorch](https://pytorch.org):**  
   - For its robust deep learning framework and support for fine-tuning transformer models.

3. **[Kaggle](https://www.kaggle.com):**  
   - For hosting datasets and fostering a community to share machine learning projects.

4. **Open-Source Contributors:**  
   - For developing and maintaining libraries like `Transformers`, `NumPy`, and `Pandas`.

5. **Mentors and Collaborators:**  
   - For guidance, support, and constructive feedback throughout this project.

---

Feel free to ⭐ this repository if you find it insightful and helpful! 😊


