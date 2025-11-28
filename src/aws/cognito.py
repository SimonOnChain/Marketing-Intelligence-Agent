"""AWS Cognito integration for user authentication."""

from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import boto3
from botocore.exceptions import ClientError

from src.config.settings import get_settings


@dataclass
class CognitoUser:
    """Represents an authenticated Cognito user."""
    username: str
    email: str | None
    groups: list[str]
    access_token: str
    id_token: str
    refresh_token: str


class CognitoAuth:
    """AWS Cognito authentication client."""

    def __init__(
        self,
        user_pool_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
    ):
        """Initialize Cognito client.

        Args:
            user_pool_id: Cognito User Pool ID
            client_id: App Client ID
            client_secret: App Client Secret (if configured)
        """
        settings = get_settings()

        self.client = boto3.client(
            "cognito-idp",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id.get_secret_value() if settings.aws_access_key_id else None,
            aws_secret_access_key=settings.aws_secret_access_key.get_secret_value() if settings.aws_secret_access_key else None,
        )

        self.user_pool_id = user_pool_id or settings.cognito_user_pool_id
        self.client_id = client_id or settings.cognito_client_id
        self.client_secret = client_secret or (
            settings.cognito_client_secret.get_secret_value()
            if settings.cognito_client_secret else None
        )

        self._enabled = bool(self.user_pool_id and self.client_id)

    @property
    def enabled(self) -> bool:
        """Check if Cognito is configured."""
        return self._enabled

    def _get_secret_hash(self, username: str) -> str | None:
        """Calculate secret hash for Cognito."""
        if not self.client_secret:
            return None

        message = username + self.client_id
        dig = hmac.new(
            self.client_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(dig).decode()

    def sign_up(
        self,
        username: str,
        password: str,
        email: str,
        attributes: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Register a new user.

        Args:
            username: Unique username
            password: User password
            email: User email
            attributes: Additional user attributes

        Returns:
            Sign-up response with user sub
        """
        if not self._enabled:
            raise RuntimeError("Cognito not configured")

        user_attributes = [{"Name": "email", "Value": email}]
        if attributes:
            for name, value in attributes.items():
                user_attributes.append({"Name": name, "Value": value})

        params = {
            "ClientId": self.client_id,
            "Username": username,
            "Password": password,
            "UserAttributes": user_attributes,
        }

        secret_hash = self._get_secret_hash(username)
        if secret_hash:
            params["SecretHash"] = secret_hash

        try:
            response = self.client.sign_up(**params)
            return {
                "user_sub": response["UserSub"],
                "confirmed": response["UserConfirmed"],
                "delivery": response.get("CodeDeliveryDetails"),
            }
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "UsernameExistsException":
                raise ValueError("Username already exists") from e
            elif error_code == "InvalidPasswordException":
                raise ValueError("Password does not meet requirements") from e
            raise

    def confirm_sign_up(self, username: str, confirmation_code: str) -> bool:
        """Confirm user registration with verification code.

        Args:
            username: Username to confirm
            confirmation_code: Code sent to user's email

        Returns:
            True if confirmation successful
        """
        if not self._enabled:
            raise RuntimeError("Cognito not configured")

        params = {
            "ClientId": self.client_id,
            "Username": username,
            "ConfirmationCode": confirmation_code,
        }

        secret_hash = self._get_secret_hash(username)
        if secret_hash:
            params["SecretHash"] = secret_hash

        try:
            self.client.confirm_sign_up(**params)
            return True
        except ClientError:
            return False

    def sign_in(self, username: str, password: str) -> CognitoUser | None:
        """Authenticate user and get tokens.

        Args:
            username: Username or email
            password: User password

        Returns:
            CognitoUser with tokens, or None if authentication fails
        """
        if not self._enabled:
            raise RuntimeError("Cognito not configured")

        auth_params = {
            "USERNAME": username,
            "PASSWORD": password,
        }

        secret_hash = self._get_secret_hash(username)
        if secret_hash:
            auth_params["SECRET_HASH"] = secret_hash

        try:
            response = self.client.initiate_auth(
                ClientId=self.client_id,
                AuthFlow="USER_PASSWORD_AUTH",
                AuthParameters=auth_params,
            )

            if "ChallengeName" in response:
                # Handle MFA or other challenges
                return None

            auth_result = response["AuthenticationResult"]

            # Get user info
            user_info = self.client.get_user(
                AccessToken=auth_result["AccessToken"]
            )

            email = None
            for attr in user_info.get("UserAttributes", []):
                if attr["Name"] == "email":
                    email = attr["Value"]
                    break

            # Get user groups
            groups = self._get_user_groups(username)

            return CognitoUser(
                username=user_info["Username"],
                email=email,
                groups=groups,
                access_token=auth_result["AccessToken"],
                id_token=auth_result["IdToken"],
                refresh_token=auth_result.get("RefreshToken", ""),
            )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ("NotAuthorizedException", "UserNotFoundException"):
                return None
            raise

    def _get_user_groups(self, username: str) -> list[str]:
        """Get groups for a user."""
        try:
            response = self.client.admin_list_groups_for_user(
                Username=username,
                UserPoolId=self.user_pool_id,
            )
            return [g["GroupName"] for g in response.get("Groups", [])]
        except ClientError:
            return []

    def refresh_tokens(self, refresh_token: str, username: str) -> dict[str, str] | None:
        """Refresh access tokens.

        Args:
            refresh_token: Refresh token from sign_in
            username: Username for secret hash

        Returns:
            New tokens or None if refresh fails
        """
        if not self._enabled:
            raise RuntimeError("Cognito not configured")

        auth_params = {"REFRESH_TOKEN": refresh_token}

        secret_hash = self._get_secret_hash(username)
        if secret_hash:
            auth_params["SECRET_HASH"] = secret_hash

        try:
            response = self.client.initiate_auth(
                ClientId=self.client_id,
                AuthFlow="REFRESH_TOKEN_AUTH",
                AuthParameters=auth_params,
            )

            auth_result = response["AuthenticationResult"]
            return {
                "access_token": auth_result["AccessToken"],
                "id_token": auth_result["IdToken"],
            }
        except ClientError:
            return None

    def sign_out(self, access_token: str) -> bool:
        """Sign out user (invalidate tokens).

        Args:
            access_token: User's access token

        Returns:
            True if sign out successful
        """
        if not self._enabled:
            raise RuntimeError("Cognito not configured")

        try:
            self.client.global_sign_out(AccessToken=access_token)
            return True
        except ClientError:
            return False

    def verify_token(self, access_token: str) -> dict[str, Any] | None:
        """Verify and decode an access token.

        Args:
            access_token: Token to verify

        Returns:
            User info if valid, None otherwise
        """
        if not self._enabled:
            raise RuntimeError("Cognito not configured")

        try:
            response = self.client.get_user(AccessToken=access_token)

            user_attrs = {}
            for attr in response.get("UserAttributes", []):
                user_attrs[attr["Name"]] = attr["Value"]

            return {
                "username": response["Username"],
                "email": user_attrs.get("email"),
                "email_verified": user_attrs.get("email_verified") == "true",
                "attributes": user_attrs,
            }
        except ClientError:
            return None

    def forgot_password(self, username: str) -> dict[str, Any]:
        """Initiate password reset flow.

        Args:
            username: Username or email

        Returns:
            Delivery details for reset code
        """
        if not self._enabled:
            raise RuntimeError("Cognito not configured")

        params = {
            "ClientId": self.client_id,
            "Username": username,
        }

        secret_hash = self._get_secret_hash(username)
        if secret_hash:
            params["SecretHash"] = secret_hash

        response = self.client.forgot_password(**params)
        return response.get("CodeDeliveryDetails", {})

    def confirm_forgot_password(
        self,
        username: str,
        confirmation_code: str,
        new_password: str,
    ) -> bool:
        """Complete password reset with confirmation code.

        Args:
            username: Username
            confirmation_code: Code from email
            new_password: New password

        Returns:
            True if password reset successful
        """
        if not self._enabled:
            raise RuntimeError("Cognito not configured")

        params = {
            "ClientId": self.client_id,
            "Username": username,
            "ConfirmationCode": confirmation_code,
            "Password": new_password,
        }

        secret_hash = self._get_secret_hash(username)
        if secret_hash:
            params["SecretHash"] = secret_hash

        try:
            self.client.confirm_forgot_password(**params)
            return True
        except ClientError:
            return False


@lru_cache(maxsize=1)
def get_cognito_auth() -> CognitoAuth | None:
    """Get cached Cognito auth client if configured."""
    settings = get_settings()

    if not hasattr(settings, "cognito_user_pool_id") or not settings.cognito_user_pool_id:
        return None

    try:
        return CognitoAuth()
    except Exception:
        return None
