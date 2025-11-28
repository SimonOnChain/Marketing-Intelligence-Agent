"""AWS Lambda handler for the Marketing Intelligence Agent API."""

from __future__ import annotations

from mangum import Mangum

from src.api.main import app

# Mangum adapter for AWS Lambda + API Gateway
handler = Mangum(app, lifespan="off")


def lambda_handler(event: dict, context: dict) -> dict:
    """AWS Lambda entry point.

    This handler wraps the FastAPI app using Mangum to make it
    compatible with AWS Lambda and API Gateway.

    Args:
        event: API Gateway event
        context: Lambda context

    Returns:
        API Gateway response
    """
    return handler(event, context)
