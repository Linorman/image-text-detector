import boto3
import os
from botocore.exceptions import NoCredentialsError

# 配置AWS访问凭证
ACCESS_KEY = ''
SECRET_KEY = ''
session = boto3.Session(aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

# 定义S3链接和本地目录
s3_bucket = 'dataset-movie-clips'
s3_prefix = 'anime-run2/'
s3_dir = ['1030587', '662638', '768744', '283566', '955666', '573730', '428707', '399700']
local_directory = ['1030587', '662638', '768744', '283566', '955666', '573730', '428707', '399700']


s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                  aws_secret_access_key=SECRET_KEY)

for i in range(len(s3_dir)):
    s3_prefix = 'anime-run2/' + s3_dir[i] + '/clips/'
    local_dir = '/home/ubuntu/image-text-detector/clips/' + local_directory[i]
    try:
        files = s3.list_objects(Bucket=s3_bucket, Prefix=s3_prefix)['Contents']
    except NoCredentialsError:
        print("No AWS credentials were found.")
        exit(1)

    os.makedirs(local_dir, exist_ok=True)

    for file in files:
        file_name = file['Key']
        if not file_name.endswith("/"):
            s3.download_file(s3_bucket, file_name, os.path.join(local_dir, os.path.basename(file_name)))
            print(f"Downloaded {file_name}")
