import torch
import numpy as np
from tqdm import tqdm


def evaluate_retrieval(model, dataloader, device):
    """
    Evaluate text-to-video retrieval performance
    
    Args:
        model: Trained model
        dataloader: DataLoader with video-text pairs
        device: torch device
    
    Returns:
        metrics: Dictionary with R@1, R@5, R@10, MedR, MeanR
        video_embeddings: All video embeddings
        text_embeddings: All text embeddings
        video_ids: List of video IDs
    """
    model.eval()
    
    video_embeddings = []
    text_embeddings = []
    video_ids = []
    
    print("Computing embeddings...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding"):
            frames = batch['frames'].to(device)
            texts = batch['description']
            
            # Get embeddings
            video_emb, text_emb = model(frames, texts)
            
            video_embeddings.append(video_emb.cpu())
            text_embeddings.append(text_emb.cpu())
            video_ids.extend(batch['video_id'])
    
    video_embeddings = torch.cat(video_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    
    # Compute similarity matrix
    print("Computing similarities...")
    similarity_matrix = video_embeddings @ text_embeddings.t()
    
    # Text-to-Video retrieval metrics
    ranks = []
    for i in range(similarity_matrix.shape[1]):
        # For each text query
        sims = similarity_matrix[:, i]
        
        # Sort in descending order
        sorted_indices = torch.argsort(sims, descending=True)
        
        # Find rank of correct video (same index)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        ranks.append(rank + 1)  # Rank starts from 1
    
    ranks = np.array(ranks)
    
    # Calculate metrics
    metrics = {
        'R@1': 100 * (ranks <= 1).mean(),
        'R@5': 100 * (ranks <= 5).mean(),
        'R@10': 100 * (ranks <= 10).mean(),
        'MedR': np.median(ranks),
        'MeanR': np.mean(ranks),
    }
    
    return metrics, video_embeddings, text_embeddings, video_ids


def text_to_video_search(query, model, video_embeddings, video_ids, device, top_k=10):
    """
    Search for videos given a text query
    
    Args:
        query: Text query string
        model: Trained model
        video_embeddings: Precomputed video embeddings
        video_ids: List of video IDs
        device: torch device
        top_k: Number of results to return
    
    Returns:
        List of (video_id, score) tuples
    """
    model.eval()
    
    with torch.no_grad():
        # Encode query
        text_emb = model.text_encoder([query])
        text_emb = text_emb.to(video_embeddings.device)
        
        # Compute similarities
        similarities = torch.matmul(video_embeddings, text_emb.T).squeeze()
        
        # Get top-k
        top_scores, top_indices = torch.topk(similarities, k=min(top_k, len(similarities)))
    
    # Return results
    results = [
        (video_ids[idx], score.item())
        for idx, score in zip(top_indices, top_scores)
    ]
    
    return results