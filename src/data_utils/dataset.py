"""
Dataset loader for infant emotion generation.
Handles JSON labels with format: {"image.jpg": 0, "image2.jpg": 1, "image3.jpg": 2}
0 = angry, 1 = crying, 2 = happy
"""

import json
import os
from typing import Any, Dict, List

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class InfantEmotionDataset(Dataset):
    """
    Shared dataset loader for both SDXL and GAN pipelines.

    JSON format: {"image_001.jpg": 0, "image_002.jpg": 1, "image_003.jpg": 2}
    0 = angry, 1 = crying, 2 = happy
    """

    LABEL_TO_EMOTION: Dict[int, str] = {
        0: "angry",
        1: "crying",
        2: "happy"
    }

    def __init__(
        self,
        data_dir: str,
        json_path: str,
        size: int = 1024,
        center_crop: bool = False
    ) -> None:
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing image files
            json_path: Path to JSON file with labels
            size: Target image size for resizing
            center_crop: Whether to center crop or random crop
        """
        self.data_dir = data_dir
        self.size = size
        self.center_crop = center_crop

        # Load JSON labels
        with open(json_path, 'r', encoding='utf-8') as file:
            self.labels = json.load(file)

        # Build sample list (combines paths, emotions, and labels into one list)
        self.samples: List[Dict[str, Any]] = []

        for img_file, label in self.labels.items():
            img_path = os.path.join(data_dir, img_file)
            if os.path.exists(img_path):
                self.samples.append({
                    "path": img_path,
                    "emotion": self.LABEL_TO_EMOTION.get(label, "unknown"),
                    "label": label,
                })

        # Print dataset statistics
        self._print_statistics()

        # Image preprocessing
        self.image_transforms = self._create_transforms()

    def _print_statistics(self) -> None:
        """Print dataset statistics."""
        emotion_counts: Dict[str, int] = {}
        for sample in self.samples:
            emotion = sample["emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        print(f"Loaded {len(self.samples)} images")
        print(f"Distribution: {emotion_counts}")

    def _create_transforms(self) -> transforms.Compose:
        """Create image transformation pipeline."""
        resize = transforms.Resize(
            self.size,
            interpolation=transforms.InterpolationMode.BILINEAR
        )
        crop = (
            transforms.CenterCrop(self.size)
            if self.center_crop
            else transforms.RandomCrop(self.size)
        )

        return transforms.Compose([
            resize,
            crop,
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get an item from the dataset.

        Args:
            index: Index of the item to retrieve

        Returns:
            Dictionary containing image, emotion, label, and image_path
        """
        sample = self.samples[index]
        image = Image.open(sample["path"]).convert("RGB")
        image = self.image_transforms(image)

        return {
            "image": image,
            "emotion": sample["emotion"],
            "label": sample["label"],
            "image_path": sample["path"],
        }
