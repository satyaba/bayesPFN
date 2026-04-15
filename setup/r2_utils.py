#!/usr/bin/env python3
"""
R2 Utilities for BayesPFN Checkpoint Management

Usage:
    python r2_utils.py upload --file <path> --key <r2_key> [--description <desc>]
    python r2_utils.py download --key <r2_key> --output <path>
    python r2_utils.py list [--prefix <prefix>]
    python r2_utils.py delete --key <r2_key>
"""

import os
import sys
import argparse
import json
import hashlib
from datetime import datetime
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

#===============================================================================
# Configuration
#===============================================================================

def load_env():
    """Load R2 config from .env file"""
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    return {
        'account_id': os.getenv('R2_ACCOUNT_ID', ''),
        'access_key_id': os.getenv('R2_ACCESS_KEY_ID', ''),
        'secret_access_key': os.getenv('R2_SECRET_ACCESS_KEY', ''),
        'bucket_name': os.getenv('R2_BUCKET_NAME', 'bayespfn-checkpoints'),
    }

def get_s3_client(config=None):
    """Get configured S3 client for R2"""
    if config is None:
        config = load_env()
    
    session = boto3.Session(
        aws_access_key_id=config['access_key_id'],
        aws_secret_access_key=config['secret_access_key'],
        region_name='auto'
    )
    
    return session.client(
        's3',
        endpoint_url=f"https://{config['account_id']}.r2.cloudflarestorage.com"
    ), config['bucket_name']

#===============================================================================
# Upload Command
#===============================================================================

def upload_file(local_path, r2_key, description="", config=None):
    """Upload a checkpoint file to R2"""
    s3, bucket = get_s3_client(config)
    local_path = Path(local_path)
    
    if not local_path.exists():
        print(f"Error: File not found: {local_path}")
        return False
    
    # Calculate checksum
    with open(local_path, 'rb') as f:
        checksum = hashlib.md5(f.read()).hexdigest()
    
    # Metadata
    metadata = {
        'uploaded_at': datetime.now().isoformat(),
        'original_filename': local_path.name,
        'file_size': local_path.stat().st_size,
        'checksum': checksum,
        'description': description,
    }
    
    # Upload
    try:
        print(f"Uploading {local_path.name} ({metadata['file_size'] / 1024**2:.1f} MB)...")
        
        with open(local_path, 'rb') as f:
            s3.put_object(
                Bucket=bucket,
                Key=r2_key,
                Body=f,
                Metadata=metadata
            )
        
        print(f"✓ Uploaded to: {r2_key}")
        print(f"  Checksum: {checksum}")
        print(f"  Uploaded at: {metadata['uploaded_at']}")
        return True
        
    except ClientError as e:
        print(f"Error uploading: {e}")
        return False

#===============================================================================
# Download Command
#===============================================================================

def download_file(r2_key, output_path, config=None):
    """Download a checkpoint from R2"""
    s3, bucket = get_s3_client(config)
    output_path = Path(output_path)
    
    try:
        print(f"Downloading {r2_key}...")
        
        response = s3.get_object(Bucket=bucket, Key=r2_key)
        content = response['Body'].read()
        
        # Create parent dirs if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(content)
        
        print(f"✓ Downloaded to: {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024**2:.1f} MB")
        
        # Print metadata if available
        if 'Metadata' in response:
            print("  Metadata:")
            for key, value in response['Metadata'].items():
                print(f"    {key}: {value}")
        
        return True
        
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"Error: Key not found in R2: {r2_key}")
        else:
            print(f"Error downloading: {e}")
        return False

#===============================================================================
# List Command
#===============================================================================

def list_files(prefix="", config=None):
    """List files in R2 with optional prefix filter"""
    s3, bucket = get_s3_client(config)
    
    try:
        if prefix:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        else:
            response = s3.list_objects_v2(Bucket=bucket)
        
        if 'Contents' not in response:
            print("No files found")
            return []
        
        files = []
        for obj in response['Contents']:
            size_mb = obj['Size'] / 1024**2
            modified = obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {obj['Key']}")
            print(f"    Size: {size_mb:.2f} MB | Modified: {modified}")
            files.append(obj)
        
        print(f"\nTotal: {len(files)} files")
        return files
        
    except ClientError as e:
        print(f"Error listing: {e}")
        return []

#===============================================================================
# Delete Command
#===============================================================================

def delete_file(r2_key, config=None):
    """Delete a file from R2"""
    s3, bucket = get_s3_client(config)
    
    try:
        s3.delete_object(Bucket=bucket, Key=r2_key)
        print(f"✓ Deleted: {r2_key}")
        return True
        
    except ClientError as e:
        print(f"Error deleting: {e}")
        return False

#===============================================================================
# Main CLI
#===============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='R2 Utilities for BayesPFN Checkpoint Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Upload a checkpoint:
    python r2_utils.py upload --file ./checkpoints/model.ckpt --key checkpoints/bayespfn_v1/model.ckpt
  
  Download a checkpoint:
    python r2_utils.py download --key checkpoints/bayespfn_v1/model.ckpt --output ./model.ckpt
  
  List all checkpoints:
    python r2_utils.py list
  
  List specific version:
    python r2_utils.py list --prefix checkpoints/bayespfn_v1/
  
  Delete a checkpoint:
    python r2_utils.py delete --key checkpoints/bayespfn_v1/model.ckpt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Upload subcommand
    upload_parser = subparsers.add_parser('upload', help='Upload file to R2')
    upload_parser.add_argument('--file', '-f', required=True, help='Local file path')
    upload_parser.add_argument('--key', '-k', required=True, help='R2 object key')
    upload_parser.add_argument('--description', '-d', default='', help='Description')
    
    # Download subcommand
    download_parser = subparsers.add_parser('download', help='Download file from R2')
    download_parser.add_argument('--key', '-k', required=True, help='R2 object key')
    download_parser.add_argument('--output', '-o', required=True, help='Output file path')
    
    # List subcommand
    list_parser = subparsers.add_parser('list', help='List files in R2')
    list_parser.add_argument('--prefix', '-p', default='', help='Prefix filter')
    
    # Delete subcommand
    delete_parser = subparsers.add_parser('delete', help='Delete file from R2')
    delete_parser.add_argument('--key', '-k', required=True, help='R2 object key')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    success = False
    if args.command == 'upload':
        success = upload_file(args.file, args.key, args.description)
    elif args.command == 'download':
        success = download_file(args.key, args.output)
    elif args.command == 'list':
        files = list_files(args.prefix)
        success = True
    elif args.command == 'delete':
        success = delete_file(args.key)

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
