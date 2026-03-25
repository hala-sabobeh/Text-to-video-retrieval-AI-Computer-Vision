import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path


class VideoTextDataset(Dataset):
    def __init__(self, annotations_path, video_dir, config):
        """
        Dataset for video-text pairs
        
        Args:
            annotations_path: Path to annotations.txt file
            video_dir: Directory containing video files
            config: Configuration object
        """
        self.data = []
        
        # Load annotations
        with open(annotations_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # FIXED: Split by first space (not tab)
                # Format: "video_id description text"
                parts = line.split(' ', 1)  # Split on first space only
                if len(parts) >= 2:
                    video_id = parts[0]
                    description = parts[1]
                    self.data.append({
                        'video_id': video_id,
                        'description': description
                    })
        
        # Apply subset limit if configured
        if config.USE_SUBSET and len(self.data) > config.MAX_SAMPLES:
            print(f"  → Using subset: {config.MAX_SAMPLES} pairs (limited from {len(self.data)})")
            self.data = self.data[:config.MAX_SAMPLES]
        
        self.video_dir = Path(video_dir)
        self.config = config
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        video_id = item['video_id']
        video_path = self.video_dir / f"{video_id}.avi"
        
        # Load video frame
        frame = self._load_frame(video_path)
        
        # Convert to tensor: [C, T, H, W]
        frame = torch.from_numpy(frame).float() / 255.0
        frame = frame.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        frame = frame.unsqueeze(1)  # [C, H, W] -> [C, 1, H, W]
        
        return {
            'frames': frame,
            'description': item['description'],
            'video_id': video_id
        }
    
    def _load_frame(self, video_path):
        """Load a single frame from video"""
        if not video_path.exists():
            # Return black frame if video doesn't exist
            return np.zeros((self.config.FRAME_SIZE, self.config.FRAME_SIZE, 3), dtype=np.uint8)
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                # Return black frame if reading fails
                return np.zeros((self.config.FRAME_SIZE, self.config.FRAME_SIZE, 3), dtype=np.uint8)
            
            # Resize and convert color
            frame = cv2.resize(frame, (self.config.FRAME_SIZE, self.config.FRAME_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            return frame
            
        except Exception as e:
            print(f"Warning: Error loading {video_path}: {e}")
            return np.zeros((self.config.FRAME_SIZE, self.config.FRAME_SIZE, 3), dtype=np.uint8)


def get_dataloader(config, mode='train'):
    """
    Create dataloader for training or testing
    
    Args:
        config: Configuration object
        mode: 'train' or 'test'
    
    Returns:
        DataLoader
    """
    dataset = VideoTextDataset(
        config.ANNOTATIONS_PATH,
        config.VIDEO_DIR,
        config
    )
    
    print(f"Loaded {len(dataset)} video-text pairs")
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(mode == 'train'),
        num_workers=config.NUM_WORKERS,  # Must be 0 on Windows
        pin_memory=False,  # False for CPU
        drop_last=(mode == 'train')  # Drop incomplete batches during training
    )
    
    return dataloader