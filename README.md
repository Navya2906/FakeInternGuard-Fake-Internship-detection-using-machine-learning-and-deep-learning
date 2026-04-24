<img width="1661" height="874" alt="image" src="https://github.com/user-attachments/assets/20397223-9483-4e1c-9858-1cdd86e5d5d3" /># FakeInternGuard-Fake-Internship-detection-using-machine-learning-and-deep-learning
THIS PROJECT USES ML AND DL MODELS SUCH AS LOGISTIC REGRESSION, XGBOOST AND BERT

The FakeInternGuard is an AI-based System. It uses Machine Learning and Deep Learning models like Logistic regression, XGBoost and BERT to predict whether an Internship description is real or fake.

Overview:

FakeInternGuard is an intelligent web-based application developed to detect whether an internship or job posting is real or fake using Machine Learning and Deep Learning models.

Many students search for internships through online job portals, social media, and websites. Some fraudulent postings misuse this demand by publishing fake opportunities that ask for money, personal details, or make unrealistic promises. FakeInternGuard helps users identify such scam postings automatically by analyzing the internship description text.

The system uses Natural Language Processing (NLP) techniques to process job descriptions and predicts authenticity using trained AI models.

Problem Statement:

Fake internship and job postings have become common on the internet. Students and job seekers may lose:

Money through registration fees or scams
Personal information
Time and effort
Trust in online platforms

Manual verification of every posting is difficult. Therefore, an automated AI-based detection system is required.

Objective:

The main objectives of FakeInternGuard are:

Detect fake internship/job postings automatically
Protect students from scams
Provide fast and accurate predictions
Compare multiple AI models
Offer an easy-to-use web interface

Key Features:

Detects Real or Fake internship postings
Supports multiple prediction models
Simple and user-friendly interface
Instant prediction results
High model accuracy
Useful for students and job seekers

Technologies Used:

Frontend
React
HTML
CSS
JavaScript
Backend
Flask
Python

Machine Learning / Deep Learning:

Logistic Regression
XGBoost
BERT Base Uncased
Libraries
Pandas
NumPy
Scikit-learn
Transformers
PyTorch

Dataset Processing:

The internship/job dataset is cleaned and prepared before training.

Preprocessing Steps
Remove null values
Normalize labels
Balance classes
Select useful text columns
Merge text columns
Clean special characters
Convert text to lowercase

Feature Extraction:

For Logistic Regression and XGBoost

Uses TF-IDF Vectorization to convert text into numerical features.

For BERT

Uses tokenization and contextual embeddings.

Models Used:

1. Logistic Regression

A simple and fast classification algorithm suitable for text data.

2. XGBoost

An advanced boosting algorithm that gives high performance and accuracy.

3. BERT Base Uncased

A transformer-based deep learning model that understands sentence context and meaning.

Model Accuracy:

Logistic Regression ---	98%
XGBoost	--- 98.5%
BERT Base Uncased ---	96%

How the System Works:

User enters internship/job description
User selects model from dropdown
Text is sent to backend
Backend preprocesses text
Selected model predicts result
Output displayed as Real or Fake

Advantages:

Automatic scam detection
Fast prediction speed
High accuracy
Easy to use
Helps students stay safe
Reduces fraud risk

Implementation:

<img width="912" height="444" alt="image" src="https://github.com/user-attachments/assets/448d4773-16e1-43e2-b9ee-8dc9c951eb82" />

Results:

UI Design:

<img width="1034" height="336" alt="image" src="https://github.com/user-attachments/assets/72d23fcd-ad83-4abb-aec5-9b9b6da24e57" />

Real and Fake predictions with diferent Input descriptions:
<img width="1038" height="563" alt="image" src="https://github.com/user-attachments/assets/c34388bf-f6e2-4232-8613-588e6a9f3686" />
<img width="1049" height="617" alt="image" src="https://github.com/user-attachments/assets/2e6ee884-cadd-4722-beee-d5abe5a80d50" />

Models Comparision with Accuracy, F1-score:
<img width="1057" height="548" alt="image" src="https://github.com/user-attachments/assets/77b457f0-cea7-4bee-aeb3-754d561702c5" />
<img width="1025" height="544" alt="image" src="https://github.com/user-attachments/assets/ba0a391e-e8b5-4f7a-85b3-f74004293ff7" />

Future Scope :

Real-time detection
Interactive dashboard
Chatbot support
Model retraining
Cloud deployment
API integration
Enhanced security


Conclusion:

FakeInternGuard is an AI-powered solution created to help students and job seekers avoid fake internship scams. By combining Machine Learning and Deep Learning models with a web-based interface, the system provides reliable and fast predictions. It is a practical project that demonstrates the real-world use of NLP and classification algorithms in cybersecurity and fraud detection.



