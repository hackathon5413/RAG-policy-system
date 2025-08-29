"""AWS S3 utilities."""

import os

import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_s3_client():
    """Get configured S3 client."""
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "ap-south-1"),
    )


def download_from_s3(url: str, target_path: str) -> bool:
    """Download a file from S3 to target path."""
    try:
        s3_client = get_s3_client()
        # Parse the S3 URL to get bucket and key
        url_parts = url.split("/")
        bucket = url_parts[2].split(".")[0]
        key = "/".join(url_parts[3:]).split("?")[0]  # Remove query parameters

        # Download using S3 client
        s3_client.download_file(bucket, key, target_path)
        return True
    except Exception:
        return False


def is_s3_url(url: str) -> bool:
    """Check if URL is an S3 URL."""
    return "s3.amazonaws.com" in url
