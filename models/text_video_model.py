import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import timm
import numpy as np


class TextEncoder(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', embedding_dim=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Get the output dimension of the pretrained model
        self.output_dim = self.model.config.hidden_size
        
        # Project to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(self.output_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, texts):
        # Tokenize
        encoded = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=77, 
            return_tensors='pt'
        ).to(next(self.model.parameters()).device)
        
        # Get embeddings
        outputs = self.model(**encoded)
        
        # Use mean pooling
        embeddings = self._mean_pooling(outputs, encoded['attention_mask'])
        
        # Project
        embeddings = self.projection(embeddings)
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class VideoEncoder(nn.Module):
    def __init__(self, embedding_dim=512, num_frames=8):
        super().__init__()
        
        # Use a pretrained 2D CNN (ResNet50)
        self.cnn = timm.create_model('resnet50', pretrained=True, num_classes=0)
        cnn_output_dim = self.cnn.num_features
        
        # Temporal aggregation
        self.temporal_pooling = nn.AdaptiveAvgPool1d(1)
        
        # Project to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(cnn_output_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, frames):
        # frames: (B, C, T, H, W)
        batch_size, channels, num_frames, height, width = frames.shape
        
        # Reshape to process all frames at once
        frames = frames.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        frames = frames.reshape(batch_size * num_frames, channels, height, width)
        
        # Extract features using CNN
        features = self.cnn(frames)  # (B*T, feature_dim)
        
        # Reshape back
        feature_dim = features.shape[1]
        features = features.reshape(batch_size, num_frames, feature_dim)
        
        # Temporal pooling
        features = features.permute(0, 2, 1)  # (B, feature_dim, T)
        features = self.temporal_pooling(features).squeeze(-1)  # (B, feature_dim)
        
        # Project
        embeddings = self.projection(features)
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class TextVideoRetrievalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.text_encoder = TextEncoder(
            model_name=config.TEXT_ENCODER,
            embedding_dim=config.EMBEDDING_DIM
        )
        
        self.video_encoder = VideoEncoder(
            embedding_dim=config.EMBEDDING_DIM,
            num_frames=config.NUM_FRAMES
        )
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, frames, texts):
        # Encode video and text
        video_embeddings = self.video_encoder(frames)
        text_embeddings = self.text_encoder(texts)
        
        return video_embeddings, text_embeddings
    
    def compute_similarity(self, video_embeddings, text_embeddings):
        # Compute cosine similarity
        logit_scale = self.logit_scale.exp()
        logits_per_video = logit_scale * video_embeddings @ text_embeddings.t()
        logits_per_text = logits_per_video.t()
        
        return logits_per_video, logits_per_text