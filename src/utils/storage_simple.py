"""
Storage Client

Unified storage interface supporting MinIO, AWS S3, and local storage.
Provides a consistent API for storing and retrieving files across different backends.
"""

import asyncio
import aiofiles
import boto3
from botocore.exceptions import ClientError
from minio import Minio
from minio.error import S3Error
import structlog
from typing import Optional, Union
import io
from pathlib import Path
from urllib.parse import urlparse
import os

from .config import get_settings

logger = structlog.get_logger()


class StorageError(Exception):
    """Custom exception for storage operations."""
    pass


class StorageClient:
    """
    Unified storage client supporting multiple backends.
    
    Backends:
    - MinIO: Self-hosted S3-compatible storage (default)
    - AWS S3: Amazon Web Services S3
    - Local: Local filesystem storage
    """
    
    def __init__(self):
        self.settings = get_settings().storage
        self._minio_client = None
        self._s3_client = None
        self._ensure_local_directory()
    
    def _ensure_local_directory(self):
        """Ensure local storage directory exists."""
        if self.settings.storage_type == "local":
            Path(self.settings.local_storage_path).mkdir(parents=True, exist_ok=True)
    
    @property
    def minio_client(self) -> Minio:
        """Get MinIO client with lazy initialization."""
        if self._minio_client is None:
            self._minio_client = Minio(
                endpoint=self.settings.minio_endpoint,
                access_key=self.settings.minio_access_key,
                secret_key=self.settings.minio_secret_key,
                secure=self.settings.minio_secure
            )
            # Ensure bucket exists
            try:
                if not self._minio_client.bucket_exists(self.settings.minio_bucket):
                    self._minio_client.make_bucket(self.settings.minio_bucket)
                    logger.info("Created MinIO bucket", bucket=self.settings.minio_bucket)
            except Exception as e:
                logger.warning("Failed to create MinIO bucket", error=str(e))
        
        return self._minio_client
    
    async def upload_bytes(
        self,
        data: bytes,
        object_name: str,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload bytes data to the configured storage backend.
        
        Args:
            data: Bytes data to upload
            object_name: Object name in storage
            content_type: MIME type of the data
            
        Returns:
            URL or path to the uploaded data
        """
        
        logger.info(
            "Uploading bytes",
            backend=self.settings.storage_type,
            object_name=object_name,
            size=len(data)
        )
        
        try:
            if self.settings.storage_type == "minio":
                return await self._upload_bytes_to_minio(data, object_name, content_type)
            elif self.settings.storage_type == "s3":
                return await self._upload_bytes_to_s3(data, object_name, content_type)
            elif self.settings.storage_type == "local":
                return await self._upload_bytes_to_local(data, object_name)
            else:
                raise StorageError(f"Unsupported storage type: {self.settings.storage_type}")
                
        except Exception as e:
            logger.error("Bytes upload failed", error=str(e))
            raise StorageError(f"Upload failed: {str(e)}")
    
    async def download_file(self, object_name: str, local_path: Optional[Union[str, Path]] = None) -> Union[str, bytes]:
        """
        Download a file from the configured storage backend.
        
        Args:
            object_name: Object name in storage
            local_path: Local path to save file (if None, returns bytes)
            
        Returns:
            Local file path or bytes data
        """
        
        logger.info(
            "Downloading file",
            backend=self.settings.storage_type,
            object_name=object_name
        )
        
        try:
            if self.settings.storage_type == "minio":
                return await self._download_from_minio(object_name, local_path)
            elif self.settings.storage_type == "s3":
                return await self._download_from_s3(object_name, local_path)
            elif self.settings.storage_type == "local":
                return await self._download_from_local(object_name, local_path)
            else:
                raise StorageError(f"Unsupported storage type: {self.settings.storage_type}")
                
        except Exception as e:
            logger.error("File download failed", error=str(e))
            raise StorageError(f"Download failed: {str(e)}")
    
    async def download_bytes(self, object_name: str) -> bytes:
        """
        Download file contents as bytes.
        
        Args:
            object_name: Object name in storage
            
        Returns:
            File contents as bytes
        """
        result = await self.download_file(object_name, local_path=None)
        if isinstance(result, bytes):
            return result
        else:
            raise StorageError(f"Expected bytes, got {type(result)}")
    
    async def delete_file(self, object_name: str) -> bool:
        """
        Delete a file from the configured storage backend.
        
        Args:
            object_name: Object name in storage
            
        Returns:
            True if deleted successfully
        """
        
        logger.info(
            "Deleting file",
            backend=self.settings.storage_type,
            object_name=object_name
        )
        
        try:
            if self.settings.storage_type == "minio":
                return await self._delete_from_minio(object_name)
            elif self.settings.storage_type == "s3":
                return await self._delete_from_s3(object_name)
            elif self.settings.storage_type == "local":
                return await self._delete_from_local(object_name)
            else:
                raise StorageError(f"Unsupported storage type: {self.settings.storage_type}")
                
        except Exception as e:
            logger.error("File deletion failed", error=str(e))
            return False
    
    def get_file_url(self, object_name: str) -> str:
        """
        Get public URL for a file.
        
        Args:
            object_name: Object name in storage
            
        Returns:
            Public URL or local path
        """
        
        if self.settings.storage_type == "minio":
            protocol = "https" if self.settings.minio_secure else "http"
            return f"{protocol}://{self.settings.minio_endpoint}/{self.settings.minio_bucket}/{object_name}"
        elif self.settings.storage_type == "s3":
            return f"https://s3.{self.settings.aws_region}.amazonaws.com/{self.settings.s3_bucket}/{object_name}"
        elif self.settings.storage_type == "local":
            return str(Path(self.settings.local_storage_path) / object_name)
        else:
            raise StorageError(f"Unsupported storage type: {self.settings.storage_type}")
    
    # MinIO implementation
    async def _upload_bytes_to_minio(self, data: bytes, object_name: str, content_type: Optional[str]) -> str:
        """Upload bytes to MinIO."""
        loop = asyncio.get_event_loop()
        
        def _upload():
            self.minio_client.put_object(
                bucket_name=self.settings.minio_bucket,
                object_name=object_name,
                data=io.BytesIO(data),
                length=len(data),
                content_type=content_type
            )
        
        await loop.run_in_executor(None, _upload)
        return self.get_file_url(object_name)
    
    async def _download_from_minio(self, object_name: str, local_path: Optional[Path]) -> Union[str, bytes]:
        """Download file from MinIO."""
        loop = asyncio.get_event_loop()
        
        if local_path:
            def _download():
                self.minio_client.fget_object(
                    bucket_name=self.settings.minio_bucket,
                    object_name=object_name,
                    file_path=str(local_path)
                )
            
            await loop.run_in_executor(None, _download)
            return str(local_path)
        else:
            def _download_data():
                response = self.minio_client.get_object(
                    bucket_name=self.settings.minio_bucket,
                    object_name=object_name
                )
                data = response.read()
                response.close()
                response.release_conn()
                return data
            
            return await loop.run_in_executor(None, _download_data)
    
    async def _delete_from_minio(self, object_name: str) -> bool:
        """Delete file from MinIO."""
        loop = asyncio.get_event_loop()
        
        def _delete():
            self.minio_client.remove_object(
                bucket_name=self.settings.minio_bucket,
                object_name=object_name
            )
        
        try:
            await loop.run_in_executor(None, _delete)
            return True
        except S3Error:
            return False
    
    # S3 implementation (placeholder methods)
    async def _upload_bytes_to_s3(self, data: bytes, object_name: str, content_type: Optional[str]) -> str:
        """Upload bytes to S3."""
        raise StorageError("S3 implementation not available in this simplified version")
    
    async def _download_from_s3(self, object_name: str, local_path: Optional[Path]) -> Union[str, bytes]:
        """Download file from S3."""
        raise StorageError("S3 implementation not available in this simplified version")
    
    async def _delete_from_s3(self, object_name: str) -> bool:
        """Delete file from S3."""
        raise StorageError("S3 implementation not available in this simplified version")
    
    # Local storage implementation
    async def _upload_bytes_to_local(self, data: bytes, object_name: str) -> str:
        """Upload bytes to local storage."""
        target_path = Path(self.settings.local_storage_path) / object_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(target_path, 'wb') as f:
            await f.write(data)
        
        return str(target_path)
    
    async def _download_from_local(self, object_name: str, local_path: Optional[Path]) -> Union[str, bytes]:
        """Download file from local storage."""
        source_path = Path(self.settings.local_storage_path) / object_name
        
        if not source_path.exists():
            raise StorageError(f"File not found: {object_name}")
        
        if local_path:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(source_path, 'rb') as src:
                async with aiofiles.open(local_path, 'wb') as dst:
                    data = await src.read()
                    await dst.write(data)
            
            return str(local_path)
        else:
            async with aiofiles.open(source_path, 'rb') as f:
                return await f.read()
    
    async def _delete_from_local(self, object_name: str) -> bool:
        """Delete file from local storage."""
        file_path = Path(self.settings.local_storage_path) / object_name
        
        try:
            if file_path.exists():
                file_path.unlink()
            return True
        except Exception:
            return False


# Global storage client instance
_storage_client: Optional[StorageClient] = None


def get_storage_client() -> StorageClient:
    """Get storage client (singleton pattern)."""
    global _storage_client
    
    if _storage_client is None:
        _storage_client = StorageClient()
    
    return _storage_client