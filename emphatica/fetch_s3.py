#!/usr/bin/env python3
"""
List or download Empatica Health Monitoring Platform data from the study S3 bucket.

Matches patterns described in Empatica Health Monitoring Platform / data-export documentation (S3 prefixes, `raw_data`, `digital_biomarkers/aggregated_per_minute`, etc.):
  - S3 Access URL from your access keys CSV, or BUCKET_NAME + PREFIX separately.

Environment variables:
  S3_ACCESS_URL   e.g. s3://empatica-us-east-1-prod-data/v2/001/
  BUCKET_NAME     optional if S3_ACCESS_URL is set
  PREFIX          optional if S3_ACCESS_URL is set (e.g. v2/001/)
  LOCAL_PATH      directory for downloads when using --download
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY   standard boto3 credentials

Raw wristband data often appears as .avro under participant_data/.../raw_data/<schema_version>/ per the doc hierarchy.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("Install boto3: pip install boto3", file=sys.stderr)
    raise


def parse_s3_url(url: str) -> tuple[str, str]:
    url = url.strip()
    if not url.startswith("s3://"):
        raise ValueError("S3_ACCESS_URL must start with s3://")
    rest = url[5:]
    parts = rest.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"
    return bucket, prefix


def get_bucket_prefix() -> tuple[str, str]:
    access = os.environ.get("S3_ACCESS_URL", "").strip()
    if access:
        return parse_s3_url(access)
    bucket = os.environ.get("BUCKET_NAME", "").strip()
    prefix = os.environ.get("PREFIX", "").strip()
    if not bucket:
        raise SystemExit("Set S3_ACCESS_URL or BUCKET_NAME (+ PREFIX).")
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"
    return bucket, prefix


def local_dest(local_root: str, s3_key: str, s3_prefix: str) -> str:
    local_root = os.path.abspath(local_root)
    rel = s3_key[len(s3_prefix) :] if s3_key.startswith(s3_prefix) else s3_key
    rel = rel.lstrip("/").replace("\\", "/")
    full = os.path.normpath(os.path.join(local_root, rel))
    if not full.startswith(local_root + os.sep) and full != local_root:
        raise ValueError(f"Unsafe key path: {s3_key!r}")
    return full


def main():
    parser = argparse.ArgumentParser(description="List or download Empatica S3 objects (boto3).")
    parser.add_argument(
        "--max-keys",
        type=int,
        default=500,
        help="Maximum objects to list or download (default 500).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download objects under PREFIX into LOCAL_PATH.",
    )
    parser.add_argument(
        "--biomarkers-only",
        action="store_true",
        help="Append digital_biomarkers/aggregated_per_minute/ to PREFIX (per-minute CSVs, not raw Avro).",
    )
    parser.add_argument(
        "--skip-raw-data",
        action="store_true",
        help="When listing/downloading, skip keys containing /raw_data/.",
    )
    args = parser.parse_args()

    bucket_name, s3_prefix = get_bucket_prefix()
    if args.biomarkers_only and "aggregated_per_minute" not in s3_prefix:
        s3_prefix = s3_prefix.rstrip("/") + "/digital_biomarkers/aggregated_per_minute/"
    client = boto3.client("s3")
    local_path = os.environ.get("LOCAL_PATH", "").strip()

    if args.download and not local_path:
        raise SystemExit("Set LOCAL_PATH to a directory for downloads.")

    paginator = client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(
        Bucket=bucket_name,
        Prefix=s3_prefix,
        PaginationConfig={"PageSize": min(1000, max(1, args.max_keys))},
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if args.skip_raw_data and "/raw_data/" in key.replace("\\", "/"):
                continue
            print(key)
            count += 1
            if args.download:
                dest = local_dest(local_path, key, s3_prefix)
                os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                try:
                    client.download_file(bucket_name, key, dest)
                except ClientError as e:
                    print(f"Error downloading {key}: {e}", file=sys.stderr)
            if count >= args.max_keys:
                break
        if count >= args.max_keys:
            break

    if count >= args.max_keys:
        print(f"(Stopped at --max-keys={args.max_keys}.)", file=sys.stderr)
    if args.download:
        print(f"Downloaded or attempted {count} objects into {local_path}")


if __name__ == "__main__":
    main()
