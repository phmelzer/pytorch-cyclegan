# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import glob
import os
import random
import time
from threading import Thread

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root, transform=None, unaligned=False):
        self.transform = transform
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, f"Sim") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, f"Real") + "/*.*"))

    def __getitem__(self, index):
        item_sim = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('L'))

        if self.unaligned:
            item_real = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('L'))
        else:
            item_real = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('L'))

        max_A = item_sim.max()
        max_B = item_real.max()

        item_sim = item_sim / max_A
        item_real = item_real / max_B

        return {"Sim": item_sim, "Real": item_real}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))