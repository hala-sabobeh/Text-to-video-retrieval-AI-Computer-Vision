print("=" * 60)
print("SYSTEM TEST STARTING")
print("=" * 60)

try:
    print("\n1. Testing PyTorch...")
    import torch
    print(f"   ✓ PyTorch version: {torch.__version__}")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    
    print("\n2. Testing OpenCV...")
    import cv2
    print(f"   ✓ OpenCV version: {cv2.__version__}")
    
    print("\n3. Testing other libraries...")
    import pandas as pd
    from transformers import AutoTokenizer
    import numpy as np
    print("   ✓ All libraries imported!")
    
    print("\n4. Checking data files...")
    import os
    if os.path.exists('data/annotations.txt'):
        print("   ✓ annotations.txt found")
    else:
        print("   ✗ annotations.txt NOT found")
    
    if os.path.exists('data/video_corpus.csv'):
        print("   ✓ video_corpus.csv found")
    else:
        print("   ✗ video_corpus.csv NOT found")
    
    if os.path.exists('data/videos'):
        videos = os.listdir('data/videos')
        avi_count = len([f for f in videos if f.endswith('.avi')])
        print(f"   ✓ Found {avi_count} video files in data/videos/")
    else:
        print("   ✗ data/videos folder NOT found")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE - Everything looks good!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

input("\nPress Enter to exit...")