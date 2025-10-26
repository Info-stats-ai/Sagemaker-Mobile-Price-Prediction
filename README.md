# Mobile Price Classification with AWS SageMaker

An end-to-end machine learning project that predicts mobile phone price ranges using AWS SageMaker and scikit-learn Random Forest Classifier.

## 📋 Project Overview

This project demonstrates how to:
- Train machine learning models using AWS SageMaker
- Utilize S3 for data storage
- Implement cost-effective training with Managed Spot Instances (69.1% savings!)
- Deploy models as real-time endpoints using AWS infrastructure

**Model**: Random Forest Classifier  
**Framework**: scikit-learn 0.23-1  
**Target**: Predict mobile phone price range (0-3) based on hardware specifications  
**Test Accuracy**: **89.25%**

## 🏗️ Architecture

```
Local Machine
    ├── Data Preprocessing (research.ipynb)
    ├── Upload to S3 (boto3)
    └── Launch Training Job
              ↓
         AWS SageMaker
              ├── Download training script & data
              ├── Train Random Forest model
              ├── Save model artifacts
              └── Upload to S3
              ↓
         Model Deployment
              └── Real-time endpoint (ml.m5.large)
```

## 📁 Project Structure

```
SageMaker/
├── research.ipynb          # Main notebook for EDA, training, deployment
├── script.py               # Training script executed by SageMaker
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── train_data.csv         # Training dataset (1,600 samples)
├── test_data.csv          # Test dataset (400 samples)
└── model.tar.gz           # Trained model (downloaded from S3)
```

## 🔧 Prerequisites

### AWS Requirements
- AWS Account with SageMaker access
- IAM User with permissions:
  - `AmazonSageMakerFullAccess`
  - `AmazonS3FullAccess`
  - IAM role creation/passing permissions
- SageMaker Execution Role (`sagemakeraccess`) with:
  - `AmazonSageMakerFullAccess`
  - S3 bucket access

### Local Environment
- Python 3.7+
- AWS CLI configured
- Required packages (see `requirements.txt`)

## 🚀 Setup Instructions

### 1. Clone and Setup Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure AWS Credentials

```bash
aws configure
```

Enter your AWS credentials:
- AWS Access Key ID: [Your IAM user access key]
- AWS Secret Access Key: [Your secret key]
- Default region: `us-east-1`
- Default output format: `json`

Verify configuration:
```bash
aws sts get-caller-identity
```

### 3. Create S3 Bucket

```bash
aws s3 mb s3://mobilepriceprediction31 --region us-east-1
```

## 📊 Dataset

**Source**: Mobile Price Classification Dataset  
**Features**: 20 hardware specifications (battery power, RAM, camera specs, etc.)  
**Target**: `price_range` (0: low, 1: medium, 2: high, 3: very high)  
**Training samples**: 1,600  
**Test samples**: 400

## 💻 Usage

### 1. Data Exploration and Preprocessing

Open `research.ipynb` and run cells for:
- Loading and exploring the dataset
- Feature analysis and correlation
- Train/test split (80/20)
- Data export to CSV

### 2. Upload Data to S3

```python
# Automated in notebook
sk_prefix = "sagemaker/mobile_price_classification/sklearncontainer"
trainpath = session.upload_data(path='train_data.csv', bucket='mobilepriceprediction31', key_prefix=sk_prefix)
testpath = session.upload_data(path='test_data.csv', bucket='mobilepriceprediction31', key_prefix=sk_prefix)
```

### 3. Launch Training Job

```python
from sagemaker.sklearn.estimator import SKLearn

FRAMEWORK_VERSION = "0.23-1"

sklearn_estimator = SKLearn(
    entry_point='script.py',
    role='arn:aws:iam::307369429660:role/sagemakeraccess',
    instance_type='ml.m5.large',
    instance_count=1,
    framework_version=FRAMEWORK_VERSION,
    base_job_name='mobile-price-classification',
    hyperparameters={
        'n_estimators': 100,
        'random_state': 42
    },
    use_spot_instances=True,  # Save up to 70% on costs!
    max_run=3600,
    max_wait=7200
)

# Start training
sklearn_estimator.fit({'train': trainpath, 'test': testpath}, wait=True)
```

### 4. Deploy Model as Endpoint

```python
from sagemaker.sklearn.model import SKLearnModel
from time import gmtime, strftime

model_name = "mobile-price-classification-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

sklearn_model = SKLearnModel(
    name=model_name,
    role='arn:aws:iam::307369429660:role/sagemakeraccess',
    model_data=sklearn_estimator.model_data,
    entry_point='script.py',
    framework_version=FRAMEWORK_VERSION,
)

endpoint_name = "mobile-price-classification-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name=endpoint_name
)
```

### 5. Make Predictions

```python
# Predict single sample
prediction = predictor.predict(X_test.iloc[0].values.reshape(1, -1))
print(prediction)  # Output: [0] (low price range)
```

### 6. Clean Up (Delete Endpoint)

```bash
# Via Python
sm_client.delete_endpoint(EndpointName=endpoint_name)

# Or via AWS CLI
aws sagemaker delete-endpoint --endpoint-name <endpoint-name>
```

## 📈 Results

### Training Job: `mobile-price-classification-2025-10-26-07-08-06-840`

- **Training Time**: 94 seconds
- **Billable Time**: 29 seconds (69.1% savings with spot instances!)
- **Instance Type**: ml.m5.large
- **Model Location**: `s3://sagemaker-us-east-1-307369429660/mobile-price-classification-2025-10-26-07-08-06-840/output/model.tar.gz`

### Model Performance

**Test Accuracy**: **89.25%**

**Classification Report**:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Low) | 0.95 | 0.96 | 0.96 | 105 |
| 1 (Medium) | 0.89 | 0.87 | 0.88 | 91 |
| 2 (High) | 0.78 | 0.87 | 0.82 | 92 |
| 3 (Very High) | 0.94 | 0.87 | 0.90 | 112 |
| **Overall** | **0.90** | **0.89** | **0.89** | **400** |

### Key Insights

- **Best Performance**: Class 0 (Low price) with 96% recall
- **Challenging Class**: Class 2 (High price) with 78% precision
- **Balanced Performance**: Weighted average F1-score of 0.89
- **RAM Feature**: Strongest correlation (0.917) with price range

## 💰 Cost Optimization

This project uses **Managed Spot Instances** to reduce training costs:

```python
use_spot_instances=True
max_run=3600        # Maximum training time (1 hour)
max_wait=7200       # Maximum wait for spot capacity (2 hours)
```

**Actual Savings**: 69.1% on this training job!  
**Training seconds**: 94  
**Billable seconds**: 29  

**Estimated Costs** (us-east-1):
- Training: ~$0.01 (with spot instances)
- Endpoint (1 hour): ~$0.115 (ml.m5.large)
- S3 Storage: Negligible for this dataset size

## 🔐 Security Best Practices

✅ Use IAM users (not root account)  
✅ Never commit AWS credentials to git  
✅ Use SageMaker execution roles for service-to-service auth  
✅ Restrict S3 bucket permissions  
✅ Regularly rotate access keys  
✅ Delete endpoints when not in use to avoid charges

## 🛠️ Troubleshooting

### Credentials Error
```bash
# Verify credentials
aws sts get-caller-identity

# Reconfigure if needed
aws configure
```

### SageMaker Training Job Fails
- Check CloudWatch logs: `/aws/sagemaker/TrainingJobs`
- Verify S3 data paths match script expectations
- Check execution role permissions
- Ensure `script.py` has no syntax errors

### Spot Instance Interrupted
- Increase `max_wait` time
- Use on-demand instances: `use_spot_instances=False`

### Endpoint Prediction Errors
- Ensure input shape matches training data (20 features)
- Verify endpoint is in "InService" status
- Check endpoint logs in CloudWatch

## 📚 References

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [AWS Cost Management](https://aws.amazon.com/aws-cost-management/)

## 📝 License

This project is for educational purposes.

## 👤 Author

Omkar Thakur

---

**Last Updated**: October 26, 2025  
**Training Job**: mobile-price-classification-2025-10-26-07-08-06-840  
**Test Accuracy**: 89.25%  
**Cost Savings**: 69.1%

---

**Note**: Remember to delete endpoints after use to avoid ongoing charges!
