"""
   Copyright (c) 2025, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from time import time


from dlio_benchmark.common.constants import MODULE_STORAGE
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType
import os

from dlio_benchmark.utils.utility import Profile

# Import Azure SDK libraries at module level for patching in tests
try:
    from azure.storage.filedatalake import DataLakeServiceClient
    from azure.identity import DefaultAzureCredential
except ImportError:
    DataLakeServiceClient = None
    DefaultAzureCredential = None

dlp = Profile(MODULE_STORAGE)


class ADLSGen2Storage(DataStorage):
    """
    Storage APIs for ADLS Gen2 (Azure Data Lake Storage Gen2).
    Uses Azure Data Lake Storage Gen2 Python SDK to interact with Azure storage.
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        super().__init__(framework)
        self.namespace = Namespace(namespace, NamespaceType.HIERARCHICAL)
        
        # Check if Azure SDK libraries are available
        if DataLakeServiceClient is None:
            raise ImportError(
                "Azure Storage libraries are required for ADLS Gen2 support. "
                "Install with: pip install azure-storage-file-datalake azure-identity"
            )
        
        # Import exception types locally as they're only used in this class
        from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
        
        # Store exception types for use in methods
        self.ResourceNotFoundError = ResourceNotFoundError
        self.ResourceExistsError = ResourceExistsError
        
        # Get storage configuration from args
        storage_options = getattr(self._args, "storage_options", {}) or {}
        
        # Support both connection string and account URL authentication
        connection_string = storage_options.get("connection_string")
        account_url = storage_options.get("account_url")
        account_name = storage_options.get("account_name")
        
        if connection_string:
            # Use connection string authentication
            self.service_client = DataLakeServiceClient.from_connection_string(connection_string)
        elif account_url:
            # Use account URL with default credential
            credential = DefaultAzureCredential()
            self.service_client = DataLakeServiceClient(account_url=account_url, credential=credential)
        elif account_name:
            # Construct account URL from account name
            account_url = f"https://{account_name}.dfs.core.windows.net"
            credential = DefaultAzureCredential()
            self.service_client = DataLakeServiceClient(account_url=account_url, credential=credential)
        else:
            raise ValueError(
                "ADLS Gen2 requires authentication configuration. "
                "Provide 'connection_string', 'account_url', or 'account_name' in storage_options."
            )
        
        # Get or create file system client for the namespace (container)
        self.file_system_client = self.service_client.get_file_system_client(file_system=self.namespace.name)

    @dlp.log
    def get_uri(self, id):
        return "abfs://" + os.path.join(self.namespace.name, id)

    @dlp.log
    def create_namespace(self, exist_ok=False):
        """
        Create the file system (container) for ADLS Gen2.
        """
        try:
            self.file_system_client.create_file_system()
            return True
        except self.ResourceExistsError:
            if exist_ok:
                return True
            raise
        except Exception as e:
            print(f"Error creating namespace '{self.namespace.name}': {e}")
            return False

    @dlp.log
    def get_namespace(self):
        """
        Get the namespace (file system/container) information.
        """
        try:
            properties = self.file_system_client.get_file_system_properties()
            return MetadataType.DIRECTORY
        except self.ResourceNotFoundError:
            return None

    @dlp.log
    def create_node(self, id, exist_ok=False):
        """
        Create a directory in ADLS Gen2.
        """
        try:
            # Parse abfs://container/path to extract just the path
            
            parsed = urlparse(id)
            if parsed.scheme == 'abfs':
                # Extract path without leading slash
                dir_path = parsed.path.lstrip('/')
            else:
                # If no scheme, use id as-is
                dir_path = id
            
            directory_client = self.file_system_client.get_directory_client(dir_path)
            directory_client.create_directory()
            return True
        except self.ResourceExistsError:
            if exist_ok:
                return True
            raise
        except Exception as e:
            print(f"Error creating node '{id}': {e}")
            return False

    @dlp.log
    def get_node(self, id=""):
        """
        Get metadata about a path (file or directory).
        """
        if not id or id == "":
            return self.get_namespace()
        
        # Parse abfs://container/path to extract just the path
        
        parsed = urlparse(id)
        if parsed.scheme == 'abfs':
            # Extract path without leading slash
            node_path = parsed.path.lstrip('/')
        else:
            # If no scheme, use id as-is
            node_path = id
        
        try:
            # Try as directory first
            directory_client = self.file_system_client.get_directory_client(node_path)
            properties = directory_client.get_directory_properties()
            if properties.get('is_directory', False):
                return MetadataType.DIRECTORY
            
            # Try as file
            file_client = self.file_system_client.get_file_client(node_path)
            properties = file_client.get_file_properties()
            return MetadataType.FILE
        except self.ResourceNotFoundError:
            return None
        except Exception as e:
            # If we can't determine, try to check if it's a file
            try:
                file_client = self.file_system_client.get_file_client(node_path)
                file_client.get_file_properties()
                return MetadataType.FILE
            except Exception:
                return None

    @dlp.log
    def walk_node(self, id, use_pattern=False):
        """
        List files and directories under a path.
        """
        try:
            # Parse abfs://container/path to extract just the path
            
            parsed = urlparse(id)
            if parsed.scheme == 'abfs':
                # Extract path without leading slash
                dir_path = parsed.path.lstrip('/')
            else:
                # If no scheme, use id as-is
                dir_path = id
            
            if not use_pattern:
                # List all items in the directory
                paths = self.file_system_client.get_paths(path=dir_path)
                result = []
                prefix_len = len(dir_path.rstrip('/') + '/') if dir_path else 0
                
                for path in paths:
                    path_name = path.name
                    # Get only immediate children (not nested)
                    if prefix_len > 0:
                        relative_path = path_name[prefix_len:]
                    else:
                        relative_path = path_name
                    
                    # Only include immediate children (no slashes in relative path)
                    if '/' not in relative_path:
                        result.append(relative_path)
                
                return result
            else:
                # Pattern matching for file extensions
                format_ext = dir_path.split(".")[-1]
                if format_ext != format_ext.lower():
                    raise Exception(f"Unknown file format {format_ext}")
                
                # List files matching the pattern
                paths = self.file_system_client.get_paths(path=os.path.dirname(dir_path))
                result = []
                
                # Match files with both lowercase and uppercase extensions
                lower_pattern = dir_path
                upper_pattern = dir_path.replace(format_ext, format_ext.upper())
                
                for path in paths:
                    path_name = path.name
                    if (path_name.endswith(format_ext) or 
                        path_name.endswith(format_ext.upper())):
                        result.append(os.path.basename(path_name))
                
                return result
        except Exception as e:
            print(f"Error walking node '{id}': {e}")
            return []

    @dlp.log
    def delete_node(self, id):
        """
        Delete a file or directory from ADLS Gen2.
        """
        try:
            # Parse abfs://container/path to extract just the path
            
            parsed = urlparse(id)
            if parsed.scheme == 'abfs':
                # Extract path without leading slash
                file_path = parsed.path.lstrip('/')
            else:
                # If no scheme, use id as-is
                file_path = id
            
            file_client = self.file_system_client.get_file_client(file_path)
            file_client.delete_file()
            return True
        except Exception as e:
            print(f"Error deleting node '{id}': {e}")
            return False

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        """
        Upload data to a file in ADLS Gen2.
        """
        try:
            # Parse abfs://container/path to extract just the path
            
            parsed = urlparse(id)
            if parsed.scheme == 'abfs':
                # Extract path without leading slash
                file_path = parsed.path.lstrip('/')
            else:
                # If no scheme, use id as-is
                file_path = id
            
            file_client = self.file_system_client.get_file_client(file_path)
            
            # Handle different data types
            if hasattr(data, 'getvalue'):
                # BytesIO or StringIO object
                data_bytes = data.getvalue()
            elif isinstance(data, bytes):
                data_bytes = data
            elif isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = str(data).encode('utf-8')
            
            if offset is not None and length is not None:
                # Partial write - append to existing file
                file_client.append_data(data_bytes, offset=offset, length=length)
                file_client.flush_data(offset + length)
            else:
                # Full write - create/overwrite file
                file_client.create_file()
                file_client.upload_data(data_bytes, overwrite=True)
            
            return True
        except Exception as e:
            print(f"Error putting data to '{id}': {e}")
            return False

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        """
        Download data from a file in ADLS Gen2.
        """
        try:
            # Parse abfs://container/path to extract just the path
            
            parsed = urlparse(id)
            if parsed.scheme == 'abfs':
                # Extract path without leading slash
                file_path = parsed.path.lstrip('/')
            else:
                # If no scheme, use id as-is
                file_path = id
            
            file_client = self.file_system_client.get_file_client(file_path)
            
            if offset is not None and length is not None:
                # Partial read
                download_stream = file_client.download_file(offset=offset, length=length)
            else:
                # Full read
                download_stream = file_client.download_file()
            
            return download_stream.readall()
        except Exception as e:
            print(f"Error getting data from '{id}': {e}")
            return None

    @dlp.log
    def isfile(self, id):
        """
        Check if the path is a file.
        """
        try:
            # Parse abfs://container/path to extract just the path
            
            parsed = urlparse(id)
            if parsed.scheme == 'abfs':
                # Extract path without leading slash
                file_path = parsed.path.lstrip('/')
            else:
                # If no scheme, use id as-is
                file_path = id
            
            file_client = self.file_system_client.get_file_client(file_path)
            properties = file_client.get_file_properties()
            # If we can get file properties and it's not a directory, it's a file
            return not properties.get('is_directory', False)
        except Exception:
            return False

    def get_basename(self, id):
        return os.path.basename(id)
