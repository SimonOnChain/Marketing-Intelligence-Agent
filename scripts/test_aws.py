"""Test AWS connection and services."""

import sys
sys.path.insert(0, ".")

def test_aws_connection():
    """Test AWS services connectivity."""
    print("=" * 50)
    print("AWS CONNECTION TEST")
    print("=" * 50)

    # 1. Check settings
    print("\n1. Checking AWS Configuration...")
    try:
        from src.config.settings import get_settings
        settings = get_settings()

        print(f"   AWS Region: {settings.aws_region or 'NOT SET'}")

        if settings.aws_access_key_id:
            key = settings.aws_access_key_id.get_secret_value()
            print(f"   Access Key: {key[:4]}...{key[-4:]} (configured)")
        else:
            print("   Access Key: NOT CONFIGURED")

        if settings.aws_secret_access_key:
            print("   Secret Key: ******** (configured)")
        else:
            print("   Secret Key: NOT CONFIGURED")

        aws_configured = bool(settings.aws_access_key_id and settings.aws_secret_access_key)

        if not aws_configured:
            print("\n   ⚠ AWS credentials not found in .env file")
            print("   Add these to your .env file:")
            print("   AWS_ACCESS_KEY_ID=your_key")
            print("   AWS_SECRET_ACCESS_KEY=your_secret")
            print("   AWS_REGION=us-east-1")
            return False

    except Exception as e:
        print(f"   ✗ Error loading settings: {e}")
        return False

    # 2. Test Comprehend
    print("\n2. Testing Amazon Comprehend...")
    try:
        from src.aws.comprehend import get_comprehend_client
        client = get_comprehend_client()

        if client and client.enabled:
            # Test sentiment analysis
            result = client.analyze_sentiment("This product is amazing! I love it.")

            if result.get("source") == "comprehend":
                print(f"   ✓ Comprehend connected!")
                print(f"   Test result: {result['sentiment']} (confidence: {max(result['scores'].values()):.2%})")
            else:
                print(f"   ⚠ Using fallback (local) - AWS might not be accessible")
                print(f"   Result: {result}")
        else:
            print("   ✗ Comprehend client not available")

    except Exception as e:
        print(f"   ✗ Comprehend error: {e}")

    # 3. Test Forecast client (just initialization)
    print("\n3. Testing Amazon Forecast client...")
    try:
        from src.aws.forecast import get_forecast_client
        client = get_forecast_client()

        if client:
            print(f"   ✓ Forecast client initialized")
            print(f"   AWS enabled: {client.enabled}")

            # Test local forecast
            import pandas as pd
            test_data = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=6, freq='M'),
                'value': [100, 120, 115, 130, 140, 150]
            })
            result = client.quick_forecast(test_data, 'date', 'value', periods=2)
            if result.get('success'):
                print(f"   ✓ Local forecast working: method={result.get('method')}")
        else:
            print("   ✗ Forecast client not available")

    except Exception as e:
        print(f"   ✗ Forecast error: {e}")

    # 4. Test Personalize client
    print("\n4. Testing Amazon Personalize client...")
    try:
        from src.aws.personalize import get_personalize_client
        client = get_personalize_client()

        if client:
            print(f"   ✓ Personalize client initialized")
            print(f"   AWS enabled: {client.enabled}")

            # Test local recommendations
            suggestions = client.get_query_suggestions("test_user", "sales")
            print(f"   ✓ Local suggestions working: {len(suggestions)} suggestions")
        else:
            print("   ✗ Personalize client not available")

    except Exception as e:
        print(f"   ✗ Personalize error: {e}")

    # 5. Test direct boto3 connection
    print("\n5. Testing direct AWS connection (STS)...")
    try:
        import boto3
        from src.config.settings import get_settings
        settings = get_settings()

        sts = boto3.client(
            'sts',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id.get_secret_value() if settings.aws_access_key_id else None,
            aws_secret_access_key=settings.aws_secret_access_key.get_secret_value() if settings.aws_secret_access_key else None,
        )

        identity = sts.get_caller_identity()
        print(f"   ✓ AWS Connected!")
        print(f"   Account: {identity['Account']}")
        print(f"   User ARN: {identity['Arn']}")

    except Exception as e:
        print(f"   ✗ AWS connection failed: {e}")
        return False

    print("\n" + "=" * 50)
    print("TEST COMPLETE")
    print("=" * 50)
    return True


if __name__ == "__main__":
    test_aws_connection()
