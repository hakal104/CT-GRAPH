import os
from dotenv import load_dotenv
import boto3
import nibabel as nib
import torch
import io
import numpy as np
from smart_open import open

def get_client():
    """
    Returns S3 client for data downloading/uploading.
    """
    
    load_dotenv()
    
    access_key= os.getenv('ACCESS_KEY')
    secret_key = os.getenv('SECRET_KEY')

    client = boto3.client('s3', endpoint_url='https://s3.kite.ume.de',
                     aws_access_key_id=access_key, aws_secret_access_key=secret_key)

    return client


def get_hf_token():
    """
    Provides hugging face token.
    """
    
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    
    return hf_token

def is_gzipped(s3_path, client):
    """
    Checks whether the file at given path is gzipped.

    Args:
        s3_path (str): Path to the NIfTI file on S3.
        client (boto3.client): S3 client to access the file.
    Returns:
        bool: Returns True if file is gzipped.
    """
    obj = client.get_object(Bucket="hamza-kalisch", Key=s3_path)
    file_header = obj['Body'].read(2)  # Read the first 2 bytes
    obj['Body'].close()
    return file_header == b'\x1f\x8b'


def load_nifti_from_s3(s3_path, client):
    """
    Loads a NIfTI image from S3, decompresses it if necessary, and returns the image data and affine matrix.

    Args:
        s3_path (str): Path to the NIfTI file on S3.
        client (boto3.client): S3 client to access the file.

    Returns:
        tuple: A tensor containing the image data and the affine matrix.
    """
    
    if is_gzipped(s3_path,client):
        with open(f's3://hamza-kalisch/{s3_path}', 'rb', transport_params={'client': client}) as s3_file:
            file_bytes = s3_file.read()
    else:
        obj = client.get_object(Bucket="hamza-kalisch", Key=s3_path)
        file_bytes = obj['Body'].read()
        obj['Body'].close()
    file_buffer = io.BytesIO(file_bytes)
    image = nib.Nifti1Image.from_bytes(file_buffer.read())
    image_data = image.get_fdata().astype(np.float32)
    image_affine = image.affine
    
    return (torch.tensor(image_data).unsqueeze(0), image_affine)  # Shape (1, D, H, W)

def load_np_from_s3(s3_path, client=None, compressed=False):
    """
    Loads a NIfTI image from S3, decompresses it if necessary, and returns the image data and affine matrix.

    Args:
        s3_path (str): Path to the NIfTI file on S3.
        client (boto3.client): S3 client to access the file.

    Returns:
        tuple: A tensor containing the image data and the affine matrix.
    """
   
    if client == None:
        image = np.load(s3_path)
    else:
        with open(f's3://hamza-kalisch/{s3_path}', 'rb', transport_params={'client': client}) as s3_file:
            file_bytes = s3_file.read()
        file_buffer = io.BytesIO(file_bytes)
        image = np.load(file_buffer, allow_pickle=True)
    
    image = torch.tensor(image['arr_0']) if compressed else torch.tensor(image)
    
    return image