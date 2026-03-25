"""
Text-to-Video Retrieval with Working Video Preview
Fixed version - opens videos in system player
"""
import torch
import webbrowser
from pathlib import Path
from config import Config
from models.text_video_model import TextVideoRetrievalModel
from utils.data_loader import get_dataloader
import time
import cv2
import base64


def get_video_thumbnail_base64(video_path, frame_number=0):
    """Get specific frame as base64 thumbnail"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # If frame_number is specified and valid, seek to that frame
        if frame_number > 0 and frame_number < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Resize
        frame = cv2.resize(frame, (320, 240))
        
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Convert to base64
        base64_data = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_data}"
        
    except Exception as e:
        return None


def create_html_preview(query, results, video_dir):
    """Create HTML with video thumbnails and open buttons"""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Video Retrieval Results</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}
        .query {{ font-size: 32px; color: #333; margin: 0 0 10px 0; font-weight: 600; }}
        .note {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
            color: #856404;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
        }}
        .card {{
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
            transition: all 0.3s;
        }}
        .card:hover {{ transform: translateY(-8px); box-shadow: 0 15px 40px rgba(0,0,0,0.3); }}
        .rank {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 18px;
            z-index: 10;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        }}
        .video-area {{
            position: relative;
            background: #000;
            min-height: 240px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
        }}
        .thumb {{ width: 100%; height: auto; display: block; }}
        .play-btn {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 80px;
            height: 80px;
            background: rgba(102, 126, 234, 0.9);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .play-icon {{
            width: 0;
            height: 0;
            border-left: 25px solid white;
            border-top: 15px solid transparent;
            border-bottom: 15px solid transparent;
            margin-left: 8px;
        }}
        .info {{ padding: 25px; }}
        .vid-id {{ font-size: 14px; color: #999; margin-bottom: 10px; font-family: monospace; }}
        .score-bar {{ margin-bottom: 15px; }}
        .score-label {{ display: flex; justify-content: space-between; margin-bottom: 8px; }}
        .score-text {{ font-size: 16px; color: #667eea; font-weight: 600; }}
        .score-val {{ font-size: 16px; color: #333; font-weight: bold; }}
        .progress {{ height: 8px; background: #e0e0e0; border-radius: 10px; overflow: hidden; }}
        .fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s;
        }}
        .desc {{
            font-size: 15px;
            color: #444;
            line-height: 1.7;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }}
        .btns {{ margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee; display: flex; gap: 10px; }}
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s;
            color: white;
        }}
        .btn-open {{ background: #667eea; }}
        .btn-open:hover {{ background: #5568d3; transform: translateY(-2px); }}
        .btn-folder {{ background: #28a745; }}
        .btn-folder:hover {{ background: #218838; transform: translateY(-2px); }}
        .no-vid {{ color: #999; text-align: center; padding: 60px 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="query">🔍 Query: "{query}"</h1>
            <p>📊 Found {len(results)} videos | ⏱️ {time.strftime('%H:%M:%S')}</p>
            <div class="note">
                💡 <strong>Click "Open Video"</strong> to play videos in your default player (VLC, Windows Media Player, etc.)
            </div>
        </div>
        <div class="grid">
"""
    
    print("\n   🎨 Generating thumbnails...")
    
    for rank, (video_id, score, desc) in enumerate(results, 1):
        video_path = video_dir / f"{video_id}.avi"
        print(f"      {rank}/{len(results)}...", end='\r')
        
        thumb = get_video_thumbnail_base64(video_path) if video_path.exists() else None
        score_pct = min(score * 100, 100)
        file_path = str(video_path.resolve()) if video_path.exists() else "Not found"
        folder = str(video_path.parent.resolve()) if video_path.exists() else ""
        
        html += f"""
            <div class="card">
                <div class="video-area" onclick="openVid{rank}()">
                    <div class="rank">#{rank}</div>
"""
        
        if thumb:
            html += f'<img src="{thumb}" class="thumb"><div class="play-btn"><div class="play-icon"></div></div>'
        else:
            html += '<div class="no-vid"><div style="font-size:48px">📹</div>Preview unavailable</div>'
        
        html += f"""
                </div>
                <div class="info">
                    <div class="vid-id">📹 {video_id}</div>
                    <div class="score-bar">
                        <div class="score-label">
                            <span class="score-text">⭐ Similarity</span>
                            <span class="score-val">{score:.4f}</span>
                        </div>
                        <div class="progress"><div class="fill" style="width:{score_pct}%"></div></div>
                    </div>
                    <div class="desc">💬 {desc}</div>
                    <div class="btns">
"""
        
        if video_path.exists():
            html += f"""
                        <button class="btn btn-open" onclick="openVid{rank}()">▶️ Open Video</button>
                        <button class="btn btn-folder" onclick="openFolder{rank}()">📁 Folder</button>
"""
        else:
            html += '<button class="btn" style="background:#ccc;cursor:not-allowed" disabled>❌ Not Found</button>'
        
        html += f"""
                    </div>
                </div>
            </div>
            <script>
                function openVid{rank}() {{
                    window.open('file:///{file_path.replace(chr(92), '/')}');
                }}
                function openFolder{rank}() {{
                    window.open('file:///{folder.replace(chr(92), '/')}');
                }}
            </script>
"""
    
    print(f"      {len(results)}/{len(results)} ✓")
    
    html += """
        </div>
    </div>
    <script>
        window.onload = function() {
            document.querySelectorAll('.fill').forEach(f => {
                const w = f.style.width;
                f.style.width = '0%';
                setTimeout(() => f.style.width = w, 100);
            });
        };
    </script>
</body>
</html>
"""
    return html


def load_or_create_embeddings(model, config):
    """Load embeddings with descriptions"""
    path = Path('video_embeddings.pth')
    
    if path.exists():
        print("   ✅ Loading embeddings...")
        data = torch.load(path, weights_only=False)
        emb = data['video_embeddings']
        ids = data['video_ids']
        desc = data.get('descriptions', [])
        
        if not desc:
            print("   📝 Loading descriptions...")
            loader = get_dataloader(config, mode='test')
            desc = []
            for batch in loader:
                desc.extend(batch['description'])
            data['descriptions'] = desc[:len(ids)]
            torch.save(data, path)
            print("   ✅ Saved!")
        
        return emb, ids, desc
    
    print("   📊 Computing embeddings...")
    loader = get_dataloader(config, mode='test')
    emb, ids, desc = [], [], []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            frames = batch['frames'].to(config.DEVICE)
            video_emb = model.video_encoder(frames)
            emb.append(video_emb.cpu())
            ids.extend(batch['video_id'])
            desc.extend(batch['description'])
            if (i + 1) % 10 == 0:
                print(f"      {i+1} batches...")
    
    emb = torch.cat(emb, 0)
    torch.save({'video_embeddings': emb, 'video_ids': ids, 'descriptions': desc}, path)
    return emb, ids, desc


def search_videos(query, model, emb, ids, desc, k=10, remove_duplicates=True):
    """Search videos"""
    with torch.no_grad():
        text_emb = model.text_encoder([query]).to(emb.device)
        sim = torch.matmul(emb, text_emb.T).squeeze()
        
        # Get more results initially to account for duplicates
        top_k = min(k * 3, len(sim))  # Get 3x more to filter duplicates
        scores, idx = torch.topk(sim, k=top_k)
    
    results = [(ids[i], scores[j].item(), desc[i] if i < len(desc) else "No description") 
               for j, i in enumerate(idx)]
    
    if remove_duplicates:
        # Keep only the highest-scoring instance of each video
        seen_videos = {}
        unique_results = []
        
        for video_id, score, description in results:
            if video_id not in seen_videos:
                seen_videos[video_id] = True
                unique_results.append((video_id, score, description))
                
                if len(unique_results) >= k:
                    break
        
        return unique_results
    
    return results[:k]


def main():
    print("="*70)
    print("TEXT-TO-VIDEO RETRIEVAL - VIDEO PLAYER VERSION")
    print("="*70)
    
    config = Config()
    
    print("\n📦 Loading model...")
    path = Path('checkpoints/best_model.pth')
    if not path.exists():
        print("   ❌ No model! Run: python train.py")
        return
    
    model = TextVideoRetrievalModel(config).to(config.DEVICE)
    ckpt = torch.load(path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("   ✅ Loaded!")
    
    print("\n📦 Loading embeddings...")
    emb, ids, desc = load_or_create_embeddings(model, config)
    print(f"   ✅ {len(ids)} videos")
    
    video_dir = Path(config.VIDEO_DIR)
    
    print("\n" + "="*70)
    print("🎬 READY! Enter queries (quit to exit)")
    print("="*70 + "\n")
    
    count = 0
    while True:
        q = input("🔍 Query: ").strip()
        if q.lower() in ['quit', 'exit', 'q']:
            break
        if not q:
            continue
        
        count += 1
        print(f"\n   🔎 Searching...")
        t = time.time()
        results = search_videos(q, model, emb, ids, desc)
        print(f"   ✅ Found {len(results)} in {(time.time()-t)*1000:.1f}ms")
        
        print("\n   📊 Results:")
        print("   " + "-"*66)
        for i, (v, s, d) in enumerate(results, 1):
            print(f"   {i:2}. {s:.4f} | {v}")
            print(f"       {d[:60]}...")
        print("   " + "-"*66)
        
        html = create_html_preview(q, results, video_dir)
        file = Path(f'results_{count}.html')
        file.write_text(html, encoding='utf-8')
        print(f"\n   💾 Saved: {file}")
        
        webbrowser.open(file.absolute().as_uri())
        print("   ✅ Opening...\n" + "="*70 + "\n")
    
    print(f"\n✅ Created {count} searches")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        # Quick mode
        config = Config()
        model = TextVideoRetrievalModel(config).to(config.DEVICE)
        ckpt = torch.load('checkpoints/best_model.pth', map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        emb, ids, desc = load_or_create_embeddings(model, config)
        q = ' '.join(sys.argv[1:])
        results = search_videos(q, model, emb, ids, desc)
        
        html = create_html_preview(q, results, Path(config.VIDEO_DIR))
        Path('results.html').write_text(html, encoding='utf-8')
        webbrowser.open(Path('results.html').absolute().as_uri())
    else:
        main()
    
    input("\n\nPress Enter to exit...")
