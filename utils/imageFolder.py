import os
import torch
from torchvision.datasets import ImageFolder
from typing import Tuple, Optional, Any

class CustomImageFolder(ImageFolder):
    """
    Simplified ImageFolder for high-cardinality classification tasks.
    Maps the last 4 characters of the filename directly to a class index.
    """
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, _ = self.samples[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)

        # Parsing the target value from filename for electoral report auditing
        basename = os.path.splitext(os.path.basename(path))[0]
        label_str = basename[-4:]
        
        if self.target_transform is not None:
            label_str = self.target_transform(label_str)

        # Cast sequence string to integer class (0 to 1000)
        label_tensor = torch.tensor(int(label_str), dtype=torch.long)
        
        return sample, label_tensor