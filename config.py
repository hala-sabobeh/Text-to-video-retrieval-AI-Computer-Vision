import torch

class Config:
    # Data paths
    ANNOTATIONS_PATH = 'data/annotations.txt'
    VIDEO_CORPUS_PATH = 'data/video_corpus.csv'
    VIDEO_DIR = 'data/videos/'
    
    # Model parameters
    TEXT_ENCODER = 'sentence-transformers/all-MiniLM-L6-v2'
    VIDEO_ENCODER = 'microsoft/xclip-base-patch32'
    EMBEDDING_DIM = 128
    
    # Training parameters
    BATCH_SIZE = 16  # FIXED: Changed from 1 to 16
    NUM_EPOCHS = 10  # FIXED: Changed from 1 to 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Video processing
    NUM_FRAMES = 1
    FRAME_SIZE = 64
    FPS = 1
    
    # Retrieval parameters
    TOP_K = [1, 5, 10]
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Checkpoints
    CHECKPOINT_DIR = 'checkpoints/'
    SAVE_INTERVAL = 5
    
    # SUBSET TRAINING
    USE_SUBSET = True
    MAX_SAMPLES = 1000
    
    # CRITICAL FOR WINDOWS - MUST BE 0
    NUM_WORKERS = 0