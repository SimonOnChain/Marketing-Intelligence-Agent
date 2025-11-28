"""Quick AWS connection test."""
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 50)
print("QUICK AWS CONNECTION TEST")
print("=" * 50)

# Get credentials
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_REGION", "eu-central-1")

print(f"\nRegion: {region}")
print(f"Access Key: {access_key[:4]}...{access_key[-4:]}" if access_key else "NOT SET")
print(f"Secret Key: {'*' * 8}" if secret_key else "NOT SET")

# Test STS (identity)
print("\n1. Testing AWS Identity (STS)...")
try:
    sts = boto3.client(
        'sts',
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    identity = sts.get_caller_identity()
    print(f"   [OK] Connected to AWS!")
    print(f"   Account: {identity['Account']}")
    print(f"   ARN: {identity['Arn']}")
except Exception as e:
    print(f"   [FAIL] Failed: {e}")

# Test Comprehend
print("\n2. Testing Amazon Comprehend...")
try:
    comprehend = boto3.client(
        'comprehend',
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    response = comprehend.detect_sentiment(
        Text="This product is amazing! I love it so much.",
        LanguageCode='en'
    )
    print(f"   [OK] Comprehend working!")
    print(f"   Sentiment: {response['Sentiment']}")
    print(f"   Positive score: {response['SentimentScore']['Positive']:.2%}")
except Exception as e:
    print(f"   [FAIL] Failed: {e}")

# Test Forecast client initialization
print("\n3. Testing Amazon Forecast client...")
try:
    forecast = boto3.client(
        'forecast',
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    # Just test that client initializes - don't need to call any API
    print(f"   [OK] Forecast client initialized")
except Exception as e:
    print(f"   [FAIL] Failed: {e}")

# Test Personalize client initialization
print("\n4. Testing Amazon Personalize client...")
try:
    personalize = boto3.client(
        'personalize',
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    print(f"   [OK] Personalize client initialized")
except Exception as e:
    print(f"   [FAIL] Failed: {e}")

print("\n" + "=" * 50)
print("TEST COMPLETE")
print("=" * 50)
