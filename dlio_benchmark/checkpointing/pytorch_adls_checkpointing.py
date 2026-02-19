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
import os
import torch
from dlio_benchmark.checkpointing.base_checkpointing import BaseCheckpointing
from dlio_benchmark.checkpointing.pytorch_checkpointing import PyTorchCheckpointing
from dlio_benchmark.utils.utility import Profile, dft_ai

from dlio_benchmark.common.constants import MODULE_CHECKPOINT

dlp = Profile(MODULE_CHECKPOINT)

# Import AzStorageCheckpoint at module level to allow test patching
try:
    from azstoragetorch import AzStorageCheckpoint
except ImportError:
    AzStorageCheckpoint = None

class PyTorchADLSCheckpointing(PyTorchCheckpointing):
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if PyTorchADLSCheckpointing.__instance is None:
            PyTorchADLSCheckpointing.__instance = PyTorchADLSCheckpointing()
        return PyTorchADLSCheckpointing.__instance

    @dft_ai.checkpoint.init
    def __init__(self):
        BaseCheckpointing.__init__(self, "ptadls")

        # Check if AzStorageCheckpoint is available
        if AzStorageCheckpoint is None:
            raise ImportError(
                "azstoragetorch is required for ADLS Gen2 checkpointing support. "
                "Install with: pip install azstoragetorch"
            )

        # Access config values from self.args (inherited from BaseCheckpointing)
        storage_options = getattr(self.args, "storage_options", {}) or {}

        # Support both connection string and account URL authentication
        connection_string = storage_options.get("connection_string")
        account_url = storage_options.get("account_url")
        account_name = storage_options.get("account_name")

        if connection_string:
            # Use connection string authentication
            os.environ["AZURE_STORAGE_CONNECTION_STRING"] = connection_string
            self.adls_checkpoint = AzStorageCheckpoint(connection_string=connection_string)
        elif account_url:
            # Use account URL with managed identity
            self.adls_checkpoint = AzStorageCheckpoint(account_url=account_url)
        elif account_name:
            # Construct account URL from account name
            account_url = f"https://{account_name}.dfs.core.windows.net"
            self.adls_checkpoint = AzStorageCheckpoint(account_url=account_url)
        else:
            raise ValueError(
                "ADLS Gen2 checkpointing requires authentication configuration. "
                "Provide 'connection_string', 'account_url', or 'account_name' in storage_options."
            )

    @dft_ai.checkpoint.capture
    def save_state(self, suffix, state, fsync = False):
        name = self.get_name(suffix)
        # Save checkpoint to ADLS Gen2
        with self.adls_checkpoint.writer(name) as writer:
            torch.save(state, writer)

    @dft_ai.checkpoint.restart
    def load_state(self, suffix, state):
        name = self.get_name(suffix)
        state = dict() # clear up
        # Load checkpoint from ADLS Gen2
        with self.adls_checkpoint.reader(name) as reader:
            state = torch.load(reader)
        self.logger.debug(f"checkpoint state loaded: {state}")
        assert(len(state.keys())>0)

    @dlp.log
    def save_checkpoint(self, epoch, step_number):
        super().save_checkpoint(epoch, step_number)

    @dlp.log
    def load_checkpoint(self, epoch, step_number):
        super().load_checkpoint(epoch, step_number)

    @dlp.log
    def finalize(self):
        super().finalize()

