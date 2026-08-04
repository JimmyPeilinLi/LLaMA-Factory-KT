# Copyright 2026 the LlamaFactory team.
#
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

import pytest
from torch import nn

from llamafactory.model import loader


def test_torch29_conv3d_guard_allows_only_frozen_modules(monkeypatch):
    monkeypatch.setattr(loader, "is_torch_version_greater_than", lambda version: version == "2.9.0")

    frozen_model = nn.Sequential(nn.Conv3d(1, 1, kernel_size=1), nn.Linear(1, 1))
    frozen_model[0].requires_grad_(False)
    loader._check_torch29_conv3d_compatibility(frozen_model)

    trainable_model = nn.Sequential(nn.Conv3d(1, 1, kernel_size=1), nn.Linear(1, 1))
    with pytest.raises(ValueError, match="trainable Conv3D"):
        loader._check_torch29_conv3d_compatibility(trainable_model)
