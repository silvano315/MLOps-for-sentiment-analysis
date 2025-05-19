import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.middlewares.logging_middleware import LoggingMiddleware
from app.api.routers import admin_router, evaluation_router, health_router, sentiment_router
from app.utils.config import get_settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()

is_huggingface = os.environ.get("SPACE_ID") is not None

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    root_path="/spaces/" + os.environ.get("SPACE_NAME", "") if is_huggingface else "",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LoggingMiddleware)

app.include_router(health_router.router, tags=["Health"])
app.include_router(evaluation_router.router, prefix="/api/v1", tags=["Evaluation"])
app.include_router(sentiment_router.router, prefix="/api/v1", tags=["Sentiment"])
app.include_router(admin_router.router, tags=["Admin"])

static_dir = Path(__file__).parent.parent.parent / "static"
static_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    """Serve the main UI page."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sentiment Analysis UI</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { margin-bottom: 20px; }
            textarea { width: 100%; height: 100px; padding: 10px; margin-bottom: 10px; }
            button { padding: 10px 15px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
            .result { margin-top: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 4px; }
            .positive { color: green; }
            .negative { color: red; }
            .neutral { color: gray; }
            .confidence { font-size: 0.9em; color: #666; }
        </style>
    </head>
    <body>
        <h1>Sentiment Analysis Demo</h1>
        
        <div class="container">
            <h2>Single Text Analysis</h2>
            <textarea id="singleText" placeholder="Enter text to analyze..."></textarea>
            <button id="analyzeSingleBtn">Analyze</button>
            <div id="singleResult" class="result" style="display: none;"></div>
        </div>
        
        <div class="container">
            <h2>Batch Analysis</h2>
            <p>Enter multiple texts, one per line:</p>
            <textarea id="batchText" placeholder="Enter multiple texts, one per line..."></textarea>
            <button id="analyzeBatchBtn">Analyze All</button>
            <div id="batchResult" class="result" style="display: none;"></div>
        </div>
        
        <script>

            document.addEventListener("DOMContentLoaded", () => {
                document.getElementById("analyzeSingleBtn").addEventListener("click", analyzeSingleText);
                document.getElementById("analyzeBatchBtn").addEventListener("click", analyzeBatchText);
            });
                    
            // Function to analyze a single text
            async function analyzeSingleText() {
                const text = document.getElementById('singleText').value.trim();
                if (!text) {
                    alert('Please enter some text to analyze');
                    return;
                }
                
                try {
                    const response = await fetch('/api/v1/sentiment', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text }),
                    });
                    
                    const result = await response.json();
                    displaySingleResult(result);
                } catch (error) {
                    alert('Error analyzing text: ' + error.message);
                }
            }
            
            // Function to display single result
            function displaySingleResult(result) {
                const resultDiv = document.getElementById('singleResult');
                resultDiv.style.display = 'block';
                
                resultDiv.innerHTML = `
                    <h3>Analysis Result</h3>
                    <p><strong>Text:</strong> ${result.text}</p>
                    <p><strong>Sentiment:</strong> <span class="${result.sentiment}">${result.sentiment}</span></p>
                    <p class="confidence">Confidence: ${(result.confidence * 100).toFixed(2)}%</p>
                    <h4>Probabilities:</h4>
                    <ul>
                        <li><span class="positive">Positive:</span> ${(result.probabilities.positive * 100).toFixed(2)}%</li>
                        <li><span class="neutral">Neutral:</span> ${(result.probabilities.neutral * 100).toFixed(2)}%</li>
                        <li><span class="negative">Negative:</span> ${(result.probabilities.negative * 100).toFixed(2)}%</li>
                    </ul>
                `;
            }
            
            // Function to analyze batch text
            async function analyzeBatchText() {
                const textArea = document.getElementById('batchText').value.trim();
                if (!textArea) {
                    alert('Please enter some texts to analyze');
                    return;
                }
                
                // Split by newlines and filter empty lines
                const texts = textArea.split("\\n").filter(text => text.trim());
                
                if (texts.length === 0) {
                    alert('Please enter valid texts separated by new lines');
                    return;
                }
                
                try {
                    const response = await fetch('/api/v1/sentiment/batch', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ texts }),
                    });
                    
                    const result = await response.json();
                    displayBatchResult(result);
                } catch (error) {
                    alert('Error analyzing texts: ' + error.message);
                }
            }
            
            // Function to display batch results
            function displayBatchResult(data) {
                const resultDiv = document.getElementById('batchResult');
                resultDiv.style.display = 'block';
                
                let html = `<h3>Batch Analysis Results</h3>`;
                
                data.results.forEach((result, index) => {
                    html += `
                        <div style="margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px solid #eee;">
                            <p><strong>Text ${index + 1}:</strong> ${result.text}</p>
                            <p><strong>Sentiment:</strong> <span class="${result.sentiment}">${result.sentiment}</span></p>
                            <p class="confidence">Confidence: ${(result.confidence * 100).toFixed(2)}%</p>
                        </div>
                    `;
                });
                
                resultDiv.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


def startup_event():
    """
    Execute operations on application startup.
    """

    logger.info("Starting up sentiment analysis API")
    start_time = time.time()

    from app.models.model_loader import get_model

    get_model()

    logger.info(f"Startup completed in {time.time() - start_time:.2f} seconds")


def shutdown_event():
    """
    Execute operations on application shutdown.
    """
    logger.info("Shutting down sentiment analysis API")


@asynccontextmanager
async def lifespan():

    startup_event()

    yield

    shutdown_event()
