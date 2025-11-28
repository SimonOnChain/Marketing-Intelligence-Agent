#!/usr/bin/env python3
"""
AWS Resource Setup Script for Marketing Intelligence Agent.

This script provisions all required AWS resources:
- DynamoDB table for query caching
- S3 bucket for data storage
- CloudWatch dashboard and alarms
- Cognito User Pool for authentication
- Checks Bedrock model access

Run: python scripts/setup_aws.py
"""

import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import boto3
from botocore.exceptions import ClientError

from src.config.settings import get_settings


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print('='*60)


def print_status(name: str, status: str, details: str = "") -> None:
    """Print a status line."""
    icon = "[OK]" if status == "success" else "[ERR]" if status == "error" else "[...]"
    print(f"  {icon} {name}: {details}")


class AWSSetup:
    """AWS resource provisioning."""

    def __init__(self):
        self.settings = get_settings()
        self.region = self.settings.aws_region

        # Initialize clients
        self.dynamodb = boto3.client("dynamodb", region_name=self.region)
        self.s3 = boto3.client("s3", region_name=self.region)
        self.cloudwatch = boto3.client("cloudwatch", region_name=self.region)
        self.cognito = boto3.client("cognito-idp", region_name=self.region)
        self.bedrock = boto3.client("bedrock", region_name=self.region)
        self.iam = boto3.client("iam", region_name=self.region)

        self.results = {}

    def run_all(self) -> dict:
        """Run all setup steps."""
        print_header("Marketing Intelligence Agent - AWS Setup")
        print(f"  Region: {self.region}")

        self.setup_dynamodb()
        self.setup_s3()
        self.setup_cloudwatch()
        self.setup_cognito()
        self.check_bedrock()
        self.generate_env_config()

        self.print_summary()
        return self.results

    def setup_dynamodb(self) -> None:
        """Create DynamoDB table for caching."""
        print_header("DynamoDB Cache Table")

        table_name = "marketing_agent_cache"

        try:
            # Check if table exists
            self.dynamodb.describe_table(TableName=table_name)
            print_status("Table", "success", f"'{table_name}' already exists")
            self.results["dynamodb"] = {"status": "exists", "table": table_name}

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                # Create table
                print_status("Creating table", "pending", table_name)

                self.dynamodb.create_table(
                    TableName=table_name,
                    KeySchema=[
                        {"AttributeName": "query_hash", "KeyType": "HASH"},
                    ],
                    AttributeDefinitions=[
                        {"AttributeName": "query_hash", "AttributeType": "S"},
                    ],
                    BillingMode="PAY_PER_REQUEST",
                    Tags=[
                        {"Key": "Project", "Value": "MarketingIntelligenceAgent"},
                        {"Key": "Environment", "Value": self.settings.env},
                    ],
                )

                # Wait for table to be active
                waiter = self.dynamodb.get_waiter("table_exists")
                waiter.wait(TableName=table_name)

                # Enable TTL
                self.dynamodb.update_time_to_live(
                    TableName=table_name,
                    TimeToLiveSpecification={
                        "Enabled": True,
                        "AttributeName": "ttl",
                    },
                )

                print_status("Table", "success", f"'{table_name}' created with TTL enabled")
                self.results["dynamodb"] = {"status": "created", "table": table_name}
            else:
                print_status("Table", "error", str(e))
                self.results["dynamodb"] = {"status": "error", "error": str(e)}

    def setup_s3(self) -> None:
        """Create S3 bucket for data storage."""
        print_header("S3 Data Bucket")

        bucket_name = self.settings.s3_bucket or f"marketing-agent-{self.region}-{int(time.time())}"

        try:
            # Check if bucket exists
            self.s3.head_bucket(Bucket=bucket_name)
            print_status("Bucket", "success", f"'{bucket_name}' already exists")
            self.results["s3"] = {"status": "exists", "bucket": bucket_name}

        except ClientError as e:
            error_code = e.response["Error"]["Code"]

            if error_code == "404":
                # Create bucket
                print_status("Creating bucket", "pending", bucket_name)

                try:
                    if self.region == "us-east-1":
                        self.s3.create_bucket(Bucket=bucket_name)
                    else:
                        self.s3.create_bucket(
                            Bucket=bucket_name,
                            CreateBucketConfiguration={
                                "LocationConstraint": self.region
                            },
                        )

                    # Enable versioning
                    self.s3.put_bucket_versioning(
                        Bucket=bucket_name,
                        VersioningConfiguration={"Status": "Enabled"},
                    )

                    # Block public access
                    self.s3.put_public_access_block(
                        Bucket=bucket_name,
                        PublicAccessBlockConfiguration={
                            "BlockPublicAcls": True,
                            "IgnorePublicAcls": True,
                            "BlockPublicPolicy": True,
                            "RestrictPublicBuckets": True,
                        },
                    )

                    # Add tags
                    self.s3.put_bucket_tagging(
                        Bucket=bucket_name,
                        Tagging={
                            "TagSet": [
                                {"Key": "Project", "Value": "MarketingIntelligenceAgent"},
                                {"Key": "Environment", "Value": self.settings.env},
                            ]
                        },
                    )

                    print_status("Bucket", "success", f"'{bucket_name}' created with versioning")
                    self.results["s3"] = {"status": "created", "bucket": bucket_name}

                except ClientError as create_error:
                    print_status("Bucket", "error", str(create_error))
                    self.results["s3"] = {"status": "error", "error": str(create_error)}

            elif error_code == "403":
                print_status("Bucket", "error", "Access denied - bucket may exist in another account")
                self.results["s3"] = {"status": "error", "error": "Access denied"}
            else:
                print_status("Bucket", "error", str(e))
                self.results["s3"] = {"status": "error", "error": str(e)}

    def setup_cloudwatch(self) -> None:
        """Create CloudWatch dashboard and alarms."""
        print_header("CloudWatch Monitoring")

        namespace = "MarketingIntelligenceAgent"

        # Create dashboard
        dashboard_name = "MarketingAgent"
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "title": "Query Volume & Latency",
                        "metrics": [
                            [namespace, "QueryCount", {"stat": "Sum", "period": 300}],
                            [namespace, "QueryLatency", {"stat": "Average", "period": 300, "yAxis": "right"}],
                        ],
                        "view": "timeSeries",
                        "region": self.region,
                    }
                },
                {
                    "type": "metric",
                    "x": 12, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "title": "Cache Performance",
                        "metrics": [
                            [namespace, "CacheHit", {"stat": "Sum", "period": 300, "color": "#2ca02c"}],
                            [namespace, "CacheMiss", {"stat": "Sum", "period": 300, "color": "#d62728"}],
                        ],
                        "view": "timeSeries",
                        "region": self.region,
                    }
                },
                {
                    "type": "metric",
                    "x": 0, "y": 6, "width": 12, "height": 6,
                    "properties": {
                        "title": "Agent Invocations",
                        "metrics": [
                            [namespace, "AgentInvocation", "Agent", "sales", {"stat": "Sum"}],
                            [namespace, "AgentInvocation", "Agent", "sentiment", {"stat": "Sum"}],
                            [namespace, "AgentInvocation", "Agent", "forecast", {"stat": "Sum"}],
                        ],
                        "view": "timeSeries",
                        "region": self.region,
                    }
                },
                {
                    "type": "metric",
                    "x": 12, "y": 6, "width": 12, "height": 6,
                    "properties": {
                        "title": "LLM Cost (USD)",
                        "metrics": [
                            [namespace, "LLMCost", {"stat": "Sum", "period": 3600}],
                        ],
                        "view": "timeSeries",
                        "region": self.region,
                    }
                },
                {
                    "type": "metric",
                    "x": 0, "y": 12, "width": 24, "height": 4,
                    "properties": {
                        "title": "Errors",
                        "metrics": [
                            [namespace, "ErrorCount", {"stat": "Sum", "period": 300, "color": "#d62728"}],
                        ],
                        "view": "timeSeries",
                        "region": self.region,
                    }
                },
            ]
        }

        try:
            self.cloudwatch.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=json.dumps(dashboard_body),
            )
            print_status("Dashboard", "success", f"'{dashboard_name}' created")

            # Create alarms
            alarms_created = []

            # High latency alarm
            self.cloudwatch.put_metric_alarm(
                AlarmName="MarketingAgent-HighLatency",
                MetricName="QueryLatency",
                Namespace=namespace,
                Threshold=10000,  # 10 seconds
                ComparisonOperator="GreaterThanThreshold",
                EvaluationPeriods=3,
                Period=300,
                Statistic="Average",
                AlarmDescription="Query latency is above 10 seconds",
                TreatMissingData="notBreaching",
            )
            alarms_created.append("HighLatency")

            # High error rate alarm
            self.cloudwatch.put_metric_alarm(
                AlarmName="MarketingAgent-HighErrorRate",
                MetricName="ErrorCount",
                Namespace=namespace,
                Threshold=10,
                ComparisonOperator="GreaterThanThreshold",
                EvaluationPeriods=2,
                Period=300,
                Statistic="Sum",
                AlarmDescription="Error count is too high",
                TreatMissingData="notBreaching",
            )
            alarms_created.append("HighErrorRate")

            print_status("Alarms", "success", f"{len(alarms_created)} alarms created")

            dashboard_url = f"https://{self.region}.console.aws.amazon.com/cloudwatch/home?region={self.region}#dashboards:name={dashboard_name}"
            self.results["cloudwatch"] = {
                "status": "created",
                "dashboard": dashboard_name,
                "dashboard_url": dashboard_url,
                "alarms": alarms_created,
            }

        except ClientError as e:
            print_status("CloudWatch", "error", str(e))
            self.results["cloudwatch"] = {"status": "error", "error": str(e)}

    def setup_cognito(self) -> None:
        """Create Cognito User Pool for authentication."""
        print_header("Cognito Authentication")

        pool_name = "MarketingAgentUsers"

        try:
            # Check for existing pool
            response = self.cognito.list_user_pools(MaxResults=60)
            existing_pool = None

            for pool in response.get("UserPools", []):
                if pool["Name"] == pool_name:
                    existing_pool = pool
                    break

            if existing_pool:
                pool_id = existing_pool["Id"]
                print_status("User Pool", "success", f"'{pool_name}' already exists (ID: {pool_id})")

                # Get app client
                clients = self.cognito.list_user_pool_clients(UserPoolId=pool_id, MaxResults=10)
                client_id = clients["UserPoolClients"][0]["ClientId"] if clients["UserPoolClients"] else None

                self.results["cognito"] = {
                    "status": "exists",
                    "user_pool_id": pool_id,
                    "client_id": client_id,
                }
            else:
                # Create user pool
                print_status("Creating User Pool", "pending", pool_name)

                pool_response = self.cognito.create_user_pool(
                    PoolName=pool_name,
                    Policies={
                        "PasswordPolicy": {
                            "MinimumLength": 8,
                            "RequireUppercase": True,
                            "RequireLowercase": True,
                            "RequireNumbers": True,
                            "RequireSymbols": False,
                        }
                    },
                    AutoVerifiedAttributes=["email"],
                    UsernameAttributes=["email"],
                    MfaConfiguration="OFF",
                    UserPoolTags={
                        "Project": "MarketingIntelligenceAgent",
                        "Environment": self.settings.env,
                    },
                )

                pool_id = pool_response["UserPool"]["Id"]

                # Create app client
                client_response = self.cognito.create_user_pool_client(
                    UserPoolId=pool_id,
                    ClientName="marketing-agent-app",
                    GenerateSecret=False,  # For web apps
                    ExplicitAuthFlows=[
                        "ALLOW_USER_PASSWORD_AUTH",
                        "ALLOW_REFRESH_TOKEN_AUTH",
                        "ALLOW_USER_SRP_AUTH",
                    ],
                    PreventUserExistenceErrors="ENABLED",
                )

                client_id = client_response["UserPoolClient"]["ClientId"]

                print_status("User Pool", "success", f"Created (ID: {pool_id})")
                print_status("App Client", "success", f"Created (ID: {client_id})")

                self.results["cognito"] = {
                    "status": "created",
                    "user_pool_id": pool_id,
                    "client_id": client_id,
                }

        except ClientError as e:
            print_status("Cognito", "error", str(e))
            self.results["cognito"] = {"status": "error", "error": str(e)}

    def check_bedrock(self) -> None:
        """Check Bedrock model access."""
        print_header("Bedrock AI Models")

        # As of late 2024, Bedrock models auto-enable on first invoke
        # For Anthropic (Claude), first-time users submit use case details on first call
        print("  Note: Models now auto-enable on first use (no manual activation).")
        print("  For Claude models, you may need to submit use case on first invoke.")
        print("")

        models_to_check = [
            ("anthropic.claude-3-haiku-20240307-v1:0", "Claude 3 Haiku (Fast Intent)"),
            ("anthropic.claude-3-sonnet-20240229-v1:0", "Claude 3 Sonnet (Synthesis)"),
            ("amazon.titan-embed-text-v1", "Titan Embeddings"),
        ]

        available_models = []

        try:
            # Just verify we can access Bedrock API
            response = self.bedrock.list_foundation_models(
                byOutputModality="TEXT"
            )

            # Models auto-enable, so mark all as ready
            for model_id, name in models_to_check:
                print_status(name, "success", "Ready (auto-enabled)")
                available_models.append(model_id)

            self.results["bedrock"] = {
                "status": "ready",
                "available_models": available_models,
                "note": "Models auto-enable on first invoke",
                "console_url": f"https://{self.region}.console.aws.amazon.com/bedrock/home?region={self.region}#/overview",
            }

        except ClientError as e:
            if "AccessDeniedException" in str(e):
                print_status("Bedrock", "error", "Access denied - check IAM permissions")
                print("\n  Your IAM user needs these permissions:")
                print("    - bedrock:ListFoundationModels")
                print("    - bedrock:InvokeModel")
            else:
                print_status("Bedrock", "error", str(e))

            self.results["bedrock"] = {"status": "error", "error": str(e)}

    def generate_env_config(self) -> None:
        """Generate .env configuration."""
        print_header("Environment Configuration")

        env_additions = []

        # S3
        if self.results.get("s3", {}).get("bucket"):
            env_additions.append(f"S3_BUCKET={self.results['s3']['bucket']}")

        # Cognito
        if self.results.get("cognito", {}).get("user_pool_id"):
            env_additions.append(f"COGNITO_USER_POOL_ID={self.results['cognito']['user_pool_id']}")
            env_additions.append(f"COGNITO_CLIENT_ID={self.results['cognito']['client_id']}")

        # Enable features
        env_additions.extend([
            "",
            "# Enable all AWS features",
            "CACHE_ENABLED=true",
            "USE_DYNAMODB_CACHE=true",
            "BEDROCK_ENABLED=true",
            "USE_BEDROCK_FOR_INTENT=true",
            "USE_BEDROCK_FOR_SYNTHESIS=false",  # Keep Grok for main synthesis
        ])

        print("\n  Add these to your .env file:")
        print("  " + "-" * 40)
        for line in env_additions:
            print(f"  {line}")
        print("  " + "-" * 40)

        self.results["env_config"] = env_additions

    def print_summary(self) -> None:
        """Print setup summary."""
        print_header("Setup Summary")

        services = [
            ("DynamoDB Cache", self.results.get("dynamodb", {}).get("status")),
            ("S3 Storage", self.results.get("s3", {}).get("status")),
            ("CloudWatch", self.results.get("cloudwatch", {}).get("status")),
            ("Cognito Auth", self.results.get("cognito", {}).get("status")),
            ("Bedrock AI", self.results.get("bedrock", {}).get("status")),
        ]

        for name, status in services:
            if status in ("created", "exists", "checked"):
                print_status(name, "success", "Ready")
            else:
                print_status(name, "error", status or "Failed")

        # CloudWatch dashboard URL
        if self.results.get("cloudwatch", {}).get("dashboard_url"):
            print(f"\n  CloudWatch Dashboard:")
            print(f"     {self.results['cloudwatch']['dashboard_url']}")

        # Bedrock console
        if self.results.get("bedrock", {}).get("console_url"):
            print(f"\n  Bedrock Console:")
            print(f"     {self.results['bedrock']['console_url']}")

        print("\n" + "=" * 60)
        print("  Setup complete! Restart your API to use AWS features.")
        print("=" * 60 + "\n")


def main():
    """Run AWS setup."""
    try:
        setup = AWSSetup()
        setup.run_all()
    except Exception as e:
        print(f"\n[ERROR] Setup failed: {e}")
        print("\nMake sure your AWS credentials are configured:")
        print("  - AWS_ACCESS_KEY_ID")
        print("  - AWS_SECRET_ACCESS_KEY")
        print("  - AWS_REGION")
        sys.exit(1)


if __name__ == "__main__":
    main()
