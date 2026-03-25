import torch
import argparse
from pathlib import Path
import time

from config import Config
from utils.data_loader import get_dataloader
from models.text_video_model import TextVideoRetrievalModel
from utils.evaluation import evaluate_retrieval, text_to_video_search


def load_model(checkpoint_path, config):
    """Load trained model from checkpoint"""
    model = TextVideoRetrievalModel(config).to(config.DEVICE)
    
    # FIX: Add weights_only=False to load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def precompute_embeddings(model, dataloader, device):
    """Precompute video embeddings for fast retrieval"""
    model.eval()
    
    video_embeddings = []
    video_ids = []
    
    print("Precomputing video embeddings...")
    with torch.no_grad():
        for batch in dataloader:
            frames = batch['frames'].to(device)
            
            # Get video embeddings only
            video_emb = model.video_encoder(frames)
            
            video_embeddings.append(video_emb.cpu())
            video_ids.extend(batch['video_id'])
    
    video_embeddings = torch.cat(video_embeddings, dim=0)
    
    # Remove duplicates (same video might have multiple captions)
    unique_video_ids = []
    unique_embeddings = []
    seen = set()
    
    for vid_id, emb in zip(video_ids, video_embeddings):
        if vid_id not in seen:
            seen.add(vid_id)
            unique_video_ids.append(vid_id)
            unique_embeddings.append(emb)
    
    unique_embeddings = torch.stack(unique_embeddings)
    
    return unique_embeddings, unique_video_ids


def interactive_search(model, video_embeddings, video_ids, config):
    """Interactive search interface"""
    print("\n" + "="*50)
    print("Text-to-Video Retrieval System")
    print("="*50)
    print("Enter your text queries. Type 'quit' to exit.")
    print("="*50 + "\n")
    
    while True:
        query = input("Enter query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        # Perform search
        start_time = time.time()
        results = text_to_video_search(
            query, 
            model, 
            video_embeddings, 
            video_ids, 
            config.DEVICE, 
            top_k=10
        )
        retrieval_time = time.time() - start_time
        
        # Display results
        print(f"\nTop 10 results (retrieved in {retrieval_time*1000:.2f}ms):")
        print("-" * 50)
        for rank, (video_id, score) in enumerate(results, 1):
            print(f"{rank}. Video ID: {video_id} | Similarity: {score:.4f}")
        print("-" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Text-to-Video Retrieval')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'evaluate'],
                       help='Run mode: interactive search or evaluation')
    parser.add_argument('--query', type=str, default=None,
                       help='Single query for non-interactive mode')
    args = parser.parse_args()
    
    # Configuration
    config = Config()
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, config)
    print("Model loaded successfully!")
    
    # Load data
    print("Loading data...")
    dataloader = get_dataloader(config, mode='test')
    
    if args.mode == 'evaluate':
        # Evaluate model
        print("Evaluating model...")
        
        # Measure retrieval time
        start_time = time.time()
        metrics, video_embeddings, text_embeddings, video_ids = evaluate_retrieval(
            model, dataloader, config.DEVICE
        )
        retrieval_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("EVALUATION RESULTS - TEXT-TO-VIDEO RETRIEVAL")
        print("="*70)
        print("\nThe performance of the text-to-video retrieval system is evaluated using:\n")
        
        # Print metrics in the format from the image
        print(f"1. Recall@K (R@K): Measures the proportion of queries for which at least one")
        print(f"   relevant video appears in the top-K retrieved results.")
        print(f"   • Recall@1:  {metrics['R@1']:.2f}%")
        print(f"   • Recall@5:  {metrics['R@5']:.2f}%")
        print(f"   • Recall@10: {metrics['R@10']:.2f}%")
        
        print(f"\n2. Median Rank (MedR): Reports the median rank of the correct video for all")
        print(f"   text queries.")
        print(f"   • Median Rank: {metrics['MedR']:.2f}")
        
        print(f"\n3. Mean Rank (MeanR): Measures the average ranking position of the relevant")
        print(f"   videos.")
        print(f"   • Mean Rank: {metrics['MeanR']:.2f}")
        
        print(f"\n4. Mean Average Precision (mAP): Evaluates ranking quality across all text")
        print(f"   queries.")
        print(f"   • mAP: {metrics.get('mAP', 0.0):.2f}%")
        
        print(f"\n5. Retrieval Time: Assesses the efficiency of the system for large-scale")
        print(f"   video databases.")
        print(f"   • Total Time: {retrieval_time:.2f}s")
        print(f"   • Time per query: {(retrieval_time / len(dataloader.dataset) * 1000):.2f}ms")
        
        print("\n" + "="*70 + "\n")
        
        # Save embeddings
        torch.save({
            'video_embeddings': video_embeddings,
            'video_ids': video_ids,
        }, 'video_embeddings.pth')
        print("Video embeddings saved to video_embeddings.pth")
        
    else:
        # Check if embeddings are already computed
        embeddings_path = 'video_embeddings.pth'
        if Path(embeddings_path).exists():
            print("Loading precomputed embeddings...")
            data = torch.load(embeddings_path, weights_only=False)
            video_embeddings = data['video_embeddings']
            video_ids = data['video_ids']
        else:
            # Precompute embeddings
            video_embeddings, video_ids = precompute_embeddings(
                model, dataloader, config.DEVICE
            )
            # Save for future use
            torch.save({
                'video_embeddings': video_embeddings,
                'video_ids': video_ids,
            }, embeddings_path)
            print(f"Video embeddings saved to {embeddings_path}")
        
        if args.query:
            # Single query mode
            results = text_to_video_search(
                args.query, 
                model, 
                video_embeddings, 
                video_ids, 
                config.DEVICE, 
                top_k=10
            )
            
            print(f"\nQuery: {args.query}")
            print("Top 10 results:")
            print("-" * 50)
            for rank, (video_id, score) in enumerate(results, 1):
                print(f"{rank}. Video ID: {video_id} | Similarity: {score:.4f}")
        else:
            # Interactive mode
            interactive_search(model, video_embeddings, video_ids, config)


if __name__ == '__main__':
    main()