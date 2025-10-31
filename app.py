#!/usr/bin/env python3
"""FastAPI web demo for DeepSeek-OCR hybrid search."""
import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from hybrid_search import TextIndex, load_st_model
from visual_index import DeepSeekVisionEmbedder, VisualIndex

# Configuration
VI_DIR = Path("./vi_index")
TI_DIR = Path("./ti_index")

# Initialize app
app = FastAPI(title="DeepSeek-OCR Hybrid Search", version="1.0.0")

# Global state for indexes and models
vi = None
tidx = None
embedder = None
st = None


def initialize_indexes():
    """Initialize indexes and models on startup."""
    global vi, tidx, embedder, st

    # Load visual index
    if VI_DIR.exists() and (VI_DIR / "hnsw.bin").exists():
        vi = VisualIndex(space="cosine")
        vi.load(VI_DIR)
        print(f"✓ Loaded visual index with {len(vi.meta)} entries")

        # Initialize embedder
        embedder = DeepSeekVisionEmbedder("deepseek-ai/DeepSeek-OCR")
        print("✓ Loaded DeepSeek vision embedder")
    else:
        print(f"⚠️  Visual index not found at {VI_DIR}")

    # Load text index
    if TI_DIR.exists() and (TI_DIR / "hnsw.bin").exists():
        tidx = TextIndex(space="cosine")
        tidx.load(TI_DIR)
        print(f"✓ Loaded text index with {len(tidx.docs)} entries")

        # Initialize sentence transformer
        st = load_st_model()
        print("✓ Loaded sentence transformer model")
    else:
        print(f"⚠️  Text index not found at {TI_DIR}")


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    initialize_indexes()


@app.get("/", response_class=HTMLResponse)
def home():
    """Home page with search forms."""
    vi_status = f"{len(vi.meta)} pages" if vi else "Not loaded"
    ti_status = f"{len(tidx.docs)} pages" if tidx else "Not loaded"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>DeepSeek-OCR Hybrid Search</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 40px auto;
                padding: 0 20px;
                line-height: 1.6;
            }}
            h1 {{
                color: #333;
                border-bottom: 2px solid #007bff;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
            }}
            .status {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .search-form {{
                background: #fff;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            input[type="text"] {{
                width: 70%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            input[type="file"] {{
                margin: 10px 0;
            }}
            input[type="number"] {{
                width: 60px;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            button {{
                background: #007bff;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
            }}
            button:hover {{
                background: #0056b3;
            }}
            .results {{
                margin-top: 20px;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 5px;
            }}
            .result-item {{
                padding: 10px;
                margin: 5px 0;
                background: white;
                border-left: 3px solid #007bff;
            }}
        </style>
        <script>
            async function searchText(event) {{
                event.preventDefault();
                const form = event.target;
                const formData = new FormData(form);

                const response = await fetch('/search_text', {{
                    method: 'POST',
                    body: formData
                }});

                const results = await response.json();
                displayResults('text-results', results, 'text');
            }}

            async function searchImage(event) {{
                event.preventDefault();
                const form = event.target;
                const formData = new FormData(form);

                const response = await fetch('/search_image', {{
                    method: 'POST',
                    body: formData
                }});

                const results = await response.json();
                displayResults('image-results', results, 'image');
            }}

            function displayResults(containerId, results, type) {{
                const container = document.getElementById(containerId);

                if (results.length === 0) {{
                    container.innerHTML = '<p>No results found.</p>';
                    return;
                }}

                let html = '<div class="results">';
                results.forEach((item, idx) => {{
                    const score = (item.score * 100).toFixed(1);
                    const name = type === 'text' ? item.name : item.display;
                    const path = type === 'text' ? `<br><small>${{item.path}}</small>` : '';

                    html += `
                        <div class="result-item">
                            <strong>${{idx + 1}}. ${{name}}</strong> (Score: ${{score}}%)
                            ${{path}}
                        </div>
                    `;
                }});
                html += '</div>';

                container.innerHTML = html;
            }}
        </script>
    </head>
    <body>
        <h1>🔍 DeepSeek-OCR Hybrid Search</h1>

        <div class="status">
            <strong>Index Status:</strong><br>
            📊 Visual Index: {vi_status}<br>
            📝 Text Index: {ti_status}
        </div>

        <h2>Text Search</h2>
        <div class="search-form">
            <form onsubmit="searchText(event)">
                <input type="text" name="q" placeholder="Enter search query..." required size="40"/>
                <label> Top K: <input type="number" name="topk" value="5" min="1" max="50"/></label>
                <button type="submit">Search</button>
            </form>
            <div id="text-results"></div>
        </div>

        <h2>Visual Search (Image Upload)</h2>
        <div class="search-form">
            <form onsubmit="searchImage(event)">
                <input type="file" name="file" accept="image/*" required/>
                <label> Top K: <input type="number" name="topk" value="5" min="1" max="50"/></label>
                <button type="submit">Search</button>
            </form>
            <div id="image-results"></div>
        </div>

        <hr style="margin-top: 40px;">
        <p style="text-align: center; color: #888;">
            <small>DeepSeek-OCR Hybrid Search Demo |
            <a href="/docs" target="_blank">API Docs</a></small>
        </p>
    </body>
    </html>
    """


@app.post("/search_text")
def search_text(q: str = Form(...), topk: int = Form(5)):
    """
    Search documents by text query.

    Args:
        q: Text query string
        topk: Number of results to return

    Returns:
        JSON array of search results
    """
    if tidx is None or st is None:
        return JSONResponse(
            {"error": "Text index not available"},
            status_code=503
        )

    try:
        qv = st.encode([q], normalize_embeddings=True)[0].astype(np.float32)
        res = tidx.query(qv, k=topk)
        return JSONResponse(
            [{"name": d["name"], "path": d.get("path", ""), "score": s} for d, s in res]
        )
    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@app.post("/search_image")
async def search_image(file: UploadFile = File(...), topk: int = Form(5)):
    """
    Search documents by visual similarity to uploaded image.

    Args:
        file: Image file upload
        topk: Number of results to return

    Returns:
        JSON array of search results
    """
    if vi is None or embedder is None:
        return JSONResponse(
            {"error": "Visual index not available"},
            status_code=503
        )

    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        qv = embedder.embed_image(img)
        res = vi.query(qv, topk=topk)
        return JSONResponse(
            [{"display": m["display"], "score": s} for m, s in res]
        )
    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "visual_index": vi is not None,
        "text_index": tidx is not None,
        "visual_entries": len(vi.meta) if vi else 0,
        "text_entries": len(tidx.docs) if tidx else 0,
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting DeepSeek-OCR Hybrid Search API...")
    print(f"Visual index: {VI_DIR}")
    print(f"Text index: {TI_DIR}")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
