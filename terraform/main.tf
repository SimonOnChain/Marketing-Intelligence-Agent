# Marketing Intelligence Agent - AWS Infrastructure
# Run: terraform init && terraform apply

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-central-1"
}

variable "environment" {
  description = "Environment (development, staging, production)"
  type        = string
  default     = "development"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "marketing-agent"
}

# Provider
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "MarketingIntelligenceAgent"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# =============================================================================
# DynamoDB - Query Cache
# =============================================================================
resource "aws_dynamodb_table" "cache" {
  name         = "${var.project_name}_cache"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "query_hash"

  attribute {
    name = "query_hash"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  point_in_time_recovery {
    enabled = true
  }

  tags = {
    Name = "${var.project_name}-cache"
  }
}

# =============================================================================
# S3 - Data Storage
# =============================================================================
resource "aws_s3_bucket" "data" {
  bucket = "${var.project_name}-data-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name = "${var.project_name}-data"
  }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# =============================================================================
# Cognito - User Authentication
# =============================================================================
resource "aws_cognito_user_pool" "main" {
  name = "${var.project_name}-users"

  username_attributes      = ["email"]
  auto_verified_attributes = ["email"]

  password_policy {
    minimum_length    = 8
    require_lowercase = true
    require_numbers   = true
    require_symbols   = false
    require_uppercase = true
  }

  account_recovery_setting {
    recovery_mechanism {
      name     = "verified_email"
      priority = 1
    }
  }

  email_configuration {
    email_sending_account = "COGNITO_DEFAULT"
  }

  schema {
    name                = "email"
    attribute_data_type = "String"
    required            = true
    mutable             = true

    string_attribute_constraints {
      min_length = 5
      max_length = 256
    }
  }

  tags = {
    Name = "${var.project_name}-user-pool"
  }
}

resource "aws_cognito_user_pool_client" "app" {
  name         = "${var.project_name}-app"
  user_pool_id = aws_cognito_user_pool.main.id

  generate_secret = false

  explicit_auth_flows = [
    "ALLOW_USER_PASSWORD_AUTH",
    "ALLOW_REFRESH_TOKEN_AUTH",
    "ALLOW_USER_SRP_AUTH",
  ]

  prevent_user_existence_errors = "ENABLED"

  access_token_validity  = 1   # hours
  id_token_validity      = 1   # hours
  refresh_token_validity = 30  # days

  token_validity_units {
    access_token  = "hours"
    id_token      = "hours"
    refresh_token = "days"
  }
}

# =============================================================================
# CloudWatch - Monitoring Dashboard
# =============================================================================
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "MarketingAgent"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Query Volume & Latency"
          region = var.aws_region
          metrics = [
            ["MarketingIntelligenceAgent", "QueryCount", { stat = "Sum", period = 300 }],
            ["MarketingIntelligenceAgent", "QueryLatency", { stat = "Average", period = 300, yAxis = "right" }]
          ]
          view = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Cache Performance"
          region = var.aws_region
          metrics = [
            ["MarketingIntelligenceAgent", "CacheHit", { stat = "Sum", period = 300, color = "#2ca02c" }],
            ["MarketingIntelligenceAgent", "CacheMiss", { stat = "Sum", period = 300, color = "#d62728" }]
          ]
          view = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        properties = {
          title  = "Agent Invocations"
          region = var.aws_region
          metrics = [
            ["MarketingIntelligenceAgent", "AgentInvocation", "Agent", "sales", { stat = "Sum" }],
            ["MarketingIntelligenceAgent", "AgentInvocation", "Agent", "sentiment", { stat = "Sum" }],
            ["MarketingIntelligenceAgent", "AgentInvocation", "Agent", "forecast", { stat = "Sum" }]
          ]
          view = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 6
        width  = 12
        height = 6
        properties = {
          title  = "LLM Cost (USD)"
          region = var.aws_region
          metrics = [
            ["MarketingIntelligenceAgent", "LLMCost", { stat = "Sum", period = 3600 }]
          ]
          view = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 12
        width  = 24
        height = 4
        properties = {
          title  = "Errors"
          region = var.aws_region
          metrics = [
            ["MarketingIntelligenceAgent", "ErrorCount", { stat = "Sum", period = 300, color = "#d62728" }]
          ]
          view = "timeSeries"
        }
      }
    ]
  })
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "high_latency" {
  alarm_name          = "${var.project_name}-high-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "QueryLatency"
  namespace           = "MarketingIntelligenceAgent"
  period              = 300
  statistic           = "Average"
  threshold           = 10000  # 10 seconds
  alarm_description   = "Query latency is above 10 seconds"
  treat_missing_data  = "notBreaching"

  tags = {
    Name = "${var.project_name}-high-latency-alarm"
  }
}

resource "aws_cloudwatch_metric_alarm" "high_error_rate" {
  alarm_name          = "${var.project_name}-high-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ErrorCount"
  namespace           = "MarketingIntelligenceAgent"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "Error count is too high"
  treat_missing_data  = "notBreaching"

  tags = {
    Name = "${var.project_name}-error-alarm"
  }
}

# =============================================================================
# Lambda - Serverless API (Optional)
# =============================================================================
resource "aws_iam_role" "lambda" {
  name = "${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_services" {
  name = "${var.project_name}-lambda-services"
  role = aws_iam_role.lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:DeleteItem",
          "dynamodb:UpdateItem",
          "dynamodb:Query",
          "dynamodb:Scan"
        ]
        Resource = aws_dynamodb_table.cache.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.data.arn,
          "${aws_s3_bucket.data.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel"
        ]
        Resource = "*"
      }
    ]
  })
}

# =============================================================================
# Outputs
# =============================================================================
output "dynamodb_table_name" {
  description = "DynamoDB cache table name"
  value       = aws_dynamodb_table.cache.name
}

output "s3_bucket_name" {
  description = "S3 data bucket name"
  value       = aws_s3_bucket.data.id
}

output "cognito_user_pool_id" {
  description = "Cognito User Pool ID"
  value       = aws_cognito_user_pool.main.id
}

output "cognito_client_id" {
  description = "Cognito App Client ID"
  value       = aws_cognito_user_pool_client.app.id
}

output "cloudwatch_dashboard_url" {
  description = "CloudWatch dashboard URL"
  value       = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=MarketingAgent"
}

output "lambda_role_arn" {
  description = "Lambda execution role ARN"
  value       = aws_iam_role.lambda.arn
}

output "env_config" {
  description = "Environment variables to add to .env"
  value       = <<-EOT

    # Add these to your .env file:
    S3_BUCKET=${aws_s3_bucket.data.id}
    COGNITO_USER_POOL_ID=${aws_cognito_user_pool.main.id}
    COGNITO_CLIENT_ID=${aws_cognito_user_pool_client.app.id}

    # Enable AWS features:
    CACHE_ENABLED=true
    USE_DYNAMODB_CACHE=true
    BEDROCK_ENABLED=true
    USE_BEDROCK_FOR_INTENT=true

  EOT
}
