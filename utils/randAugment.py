from torchvision import transforms as T
from typing import Dict, Any

class MyRandAugment(T.RandAugment):
    """
    Custom RandAugment strategy optimized for Handwritten Digit Sequence Recognition.
    
    Operations are constrained to preserve the vertical structural integrity 
    and aspect ratio of multi-digit sequences, focusing on horizontal 
    invariance and pixel-level noise.
    """
    def _augmentation_space(self, num_bins: int, image_size: Any) -> Dict[str, Any]:
        # Retrieve default augmentation space from parent class
        ops = super()._augmentation_space(num_bins, image_size)
        
        # Define valid operations that do not distort the digit sequence topology.
        # Vertical shears and translations are omitted to maintain 
        # horizontal baseline resolution.
        valid_ops = [
            "AutoContrast",
            "Equalize",
            "Posterize",
            "Sharpness",
            "ShearX",      # Horizontal shear only (preserving digit inclination)
            "TranslateX",  # Horizontal translation (sequence invariant)
            "Brightness",
            "Contrast",
            "Color"        # Controlled saturation for ink/paper contrast
        ]
        
        # Filter the operation space
        ops = {k: v for k, v in ops.items() if k in valid_ops}
        
        return ops