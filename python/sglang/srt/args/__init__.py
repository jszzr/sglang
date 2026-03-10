# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Domain-specific argument groups for SGLang server configuration."""

from sglang.srt.args.cuda_graph_args import CudaGraphArgs
from sglang.srt.args.model_args import ModelArgs
from sglang.srt.args.parallel_args import ParallelArgs

__all__ = [
    "CudaGraphArgs",
    "ModelArgs",
    "ParallelArgs",
]
