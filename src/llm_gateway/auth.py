from fastapi import Header, HTTPException, status


def verify_bearer_token(authorization: str | None = Header(None)) -> None:
    """
    Verify bearer token from Authorization header.

    Args:
        authorization: Authorization header value

    Raises:
        HTTPException: 401 if token is invalid or missing
    """
    # Import settings inside function to allow tests to set env vars first
    from llm_gateway.config import settings

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected 'Bearer <token>'",
        )

    token = authorization[7:]  # Remove "Bearer " prefix

    if token not in settings.gateway_auth_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
