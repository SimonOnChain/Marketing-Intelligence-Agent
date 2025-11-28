"""AWS S3 integration for data storage and retrieval."""

from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from src.config.settings import get_settings


class S3DataStore:
    """S3-backed data store for parquet files and other assets."""

    def __init__(self, bucket_name: str | None = None):
        """Initialize S3 client.

        Args:
            bucket_name: S3 bucket name (defaults to settings)
        """
        settings = get_settings()

        self.s3 = boto3.client(
            "s3",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id.get_secret_value() if settings.aws_access_key_id else None,
            aws_secret_access_key=settings.aws_secret_access_key.get_secret_value() if settings.aws_secret_access_key else None,
        )

        self.bucket = bucket_name or getattr(settings, "s3_bucket", None) or "marketing-agent-data"
        self._enabled = bool(settings.aws_access_key_id)

    @property
    def enabled(self) -> bool:
        """Check if S3 is available."""
        return self._enabled

    def ensure_bucket_exists(self) -> bool:
        """Create bucket if it doesn't exist."""
        if not self._enabled:
            return False

        try:
            self.s3.head_bucket(Bucket=self.bucket)
            return True
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                # Bucket doesn't exist, create it
                settings = get_settings()
                try:
                    if settings.aws_region == "us-east-1":
                        self.s3.create_bucket(Bucket=self.bucket)
                    else:
                        self.s3.create_bucket(
                            Bucket=self.bucket,
                            CreateBucketConfiguration={
                                "LocationConstraint": settings.aws_region
                            },
                        )
                    return True
                except ClientError:
                    return False
            return False

    def upload_dataframe(
        self,
        df: pd.DataFrame,
        key: str,
        format: str = "parquet",
    ) -> bool:
        """Upload a DataFrame to S3.

        Args:
            df: DataFrame to upload
            key: S3 object key (path)
            format: File format (parquet or csv)

        Returns:
            True if upload successful
        """
        if not self._enabled:
            return False

        buffer = io.BytesIO()

        if format == "parquet":
            df.to_parquet(buffer, index=False)
        elif format == "csv":
            df.to_csv(buffer, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        buffer.seek(0)

        try:
            self.s3.upload_fileobj(buffer, self.bucket, key)
            return True
        except ClientError:
            return False

    def download_dataframe(
        self,
        key: str,
        format: str = "parquet",
    ) -> pd.DataFrame | None:
        """Download a DataFrame from S3.

        Args:
            key: S3 object key
            format: File format (parquet or csv)

        Returns:
            DataFrame or None if not found
        """
        if not self._enabled:
            return None

        buffer = io.BytesIO()

        try:
            self.s3.download_fileobj(self.bucket, key, buffer)
            buffer.seek(0)

            if format == "parquet":
                return pd.read_parquet(buffer)
            elif format == "csv":
                return pd.read_csv(buffer)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except ClientError:
            return None

    def upload_file(self, local_path: str | Path, key: str) -> bool:
        """Upload a local file to S3.

        Args:
            local_path: Path to local file
            key: S3 object key

        Returns:
            True if upload successful
        """
        if not self._enabled:
            return False

        try:
            self.s3.upload_file(str(local_path), self.bucket, key)
            return True
        except ClientError:
            return False

    def download_file(self, key: str, local_path: str | Path) -> bool:
        """Download a file from S3 to local path.

        Args:
            key: S3 object key
            local_path: Local destination path

        Returns:
            True if download successful
        """
        if not self._enabled:
            return False

        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.bucket, key, str(local_path))
            return True
        except ClientError:
            return False

    def upload_json(self, data: dict | list, key: str) -> bool:
        """Upload JSON data to S3.

        Args:
            data: JSON-serializable data
            key: S3 object key

        Returns:
            True if upload successful
        """
        if not self._enabled:
            return False

        try:
            body = json.dumps(data, indent=2)
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body.encode("utf-8"),
                ContentType="application/json",
            )
            return True
        except ClientError:
            return False

    def download_json(self, key: str) -> dict | list | None:
        """Download JSON data from S3.

        Args:
            key: S3 object key

        Returns:
            Parsed JSON data or None
        """
        if not self._enabled:
            return None

        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            body = response["Body"].read().decode("utf-8")
            return json.loads(body)
        except ClientError:
            return None

    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> list[dict]:
        """List objects in the bucket.

        Args:
            prefix: Filter by key prefix
            max_keys: Maximum number of keys to return

        Returns:
            List of object metadata dicts
        """
        if not self._enabled:
            return []

        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=max_keys,
            )

            return [
                {
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                }
                for obj in response.get("Contents", [])
            ]
        except ClientError:
            return []

    def delete_object(self, key: str) -> bool:
        """Delete an object from S3.

        Args:
            key: S3 object key

        Returns:
            True if deletion successful
        """
        if not self._enabled:
            return False

        try:
            self.s3.delete_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def get_presigned_url(self, key: str, expiration: int = 3600) -> str | None:
        """Generate a presigned URL for temporary access.

        Args:
            key: S3 object key
            expiration: URL expiration in seconds

        Returns:
            Presigned URL or None
        """
        if not self._enabled:
            return None

        try:
            url = self.s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expiration,
            )
            return url
        except ClientError:
            return None

    def sync_local_data(self, local_dir: str | Path, s3_prefix: str = "data/") -> dict[str, Any]:
        """Sync local data directory to S3.

        Args:
            local_dir: Local directory path
            s3_prefix: S3 key prefix

        Returns:
            Sync results with uploaded/failed counts
        """
        if not self._enabled:
            return {"enabled": False}

        local_path = Path(local_dir)
        if not local_path.exists():
            return {"error": f"Local directory not found: {local_dir}"}

        results = {"uploaded": [], "failed": [], "skipped": []}

        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                relative = file_path.relative_to(local_path)
                s3_key = f"{s3_prefix}{relative}".replace("\\", "/")

                if self.upload_file(file_path, s3_key):
                    results["uploaded"].append(s3_key)
                else:
                    results["failed"].append(s3_key)

        return results


@lru_cache(maxsize=1)
def get_s3_store() -> S3DataStore | None:
    """Get cached S3 data store if configured."""
    settings = get_settings()

    if not settings.aws_access_key_id:
        return None

    try:
        return S3DataStore()
    except Exception:
        return None
