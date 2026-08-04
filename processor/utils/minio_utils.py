import json
import logging

from minio import Minio
from configs.minio_config import minio_config
from processor.import_processor.base import setup_logging

setup_logging()

try:
    client = Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        secure=False
    )
    if not client.bucket_exists(minio_config.bucket_name):
        client.make_bucket(minio_config.bucket_name)

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{minio_config.bucket_name}/*",
            }
        ]
    }

    client.set_bucket_policy(minio_config.bucket_name, json.dumps(policy))

except Exception as e:
    logging.error(f"Minio连接失败,错误原因{e}")
    client = None


def get_minio_client():
    return client

if __name__ == "__main__":
    get_minio_client()