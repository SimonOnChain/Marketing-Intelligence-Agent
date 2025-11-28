"""AWS service integrations."""

from src.aws.cache import QueryCache
from src.aws.bedrock import BedrockClient
from src.aws.cloudwatch import CloudWatchMetrics
from src.aws.cognito import CognitoAuth
from src.aws.s3 import S3DataStore
from src.aws.comprehend import ComprehendClient, get_comprehend_client
from src.aws.forecast import ForecastClient, get_forecast_client
from src.aws.personalize import PersonalizeClient, get_personalize_client

__all__ = [
    "QueryCache",
    "BedrockClient",
    "CloudWatchMetrics",
    "CognitoAuth",
    "S3DataStore",
    "ComprehendClient",
    "get_comprehend_client",
    "ForecastClient",
    "get_forecast_client",
    "PersonalizeClient",
    "get_personalize_client",
]
