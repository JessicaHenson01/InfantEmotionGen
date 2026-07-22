import json
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class InfantEmotionDataset(Dataset):
    """
    Shared dataset loader for both SDXL and GAN pipelines.
    
    JSON format: {"image_001.jpg": 0, "image_002.jpg": 1, "image_003.jpg": 2}
    0 = angry, 1 = crying, 2 = happy
    """
    
    def __init__(self, data_dir, json_path, size=1024, center_crop=False):
        self.data_dir = data_dir
        self.size = size
        self.center_crop = center_crop
        
        # Label mapping
        self.label_to_emotion = {
            0: "angry",
            1: "crying",
            2: "happy"
        }
        self.emotion_to_label = {v: k for k, v in self.label_to_emotion.items()}
        
        # Load JSON labels
        with open(json_path, 'r') as f:
            self.labels = json.load(f)
        
        # Build image list
        self.image_paths = []
        self.emotions = []
        self.labels_list = []
        
        for img_file, label in self.labels.items():
            img_path = os.path.join(data_dir, img_file)
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.labels_list.append(label)
                self.emotions.append(self.label_to_emotion[label])
        
        self.num_images = len(self.image_paths)
        
        # Print dataset statistics
        emotion_counts = {}
        for emotion in self.emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        print(f"Loaded {self.num_images} images")
        print(f"Distribution: {emotion_counts}")
        
        # Image preprocessing
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")
        image = self.image_transforms(image)
        
        return {
            "image": image,
            "emotion": self.emotions[index],
            "label": self.labels_list[index],
            "image_path": img_path,
        }