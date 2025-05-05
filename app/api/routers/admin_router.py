# app/api/routers/admin_router.py (versione minima)
from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse
from prometheus_client import REGISTRY, generate_latest, CONTENT_TYPE_LATEST

router = APIRouter(prefix="/admin", tags=["Admin"])

@router.get(
    "/metrics", 
    summary="Raw Prometheus metrics",
    description="Get raw Prometheus metrics"
)
async def metrics():
    """Return raw Prometheus metrics."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

@router.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Simple metrics dashboard",
    description="View a simple dashboard of sentiment analysis metrics"
)
async def simple_dashboard():
    """Simple HTML dashboard for metrics."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sentiment Analysis Metrics</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1, h2 { color: #333; }
            .metric-group { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
            table { width: 100%; border-collapse: collapse; }
            th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            .refresh-btn { padding: 10px 15px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
            #timestamp { color: #666; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h1>Sentiment Analysis Metrics Dashboard</h1>
        <p id="timestamp">Last updated: <span id="update-time">-</span></p>
        <button class="refresh-btn" onclick="fetchMetrics()">Refresh Metrics</button>
        
        <div class="metric-group">
            <h2>Sentiment Predictions</h2>
            <table id="sentiment-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Loading...</td>
                    <td></td>
                </tr>
            </table>
        </div>
        
        <div class="metric-group">
            <h2>Performance</h2>
            <table id="performance-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Loading...</td>
                    <td></td>
                </tr>
            </table>
        </div>
        
        <div class="metric-group">
            <h2>Text Metrics</h2>
            <table id="text-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Loading...</td>
                    <td></td>
                </tr>
            </table>
        </div>
        
        <script>
            // Function to parse Prometheus metrics text
            function parsePrometheusMetrics(metricsText) {
                const metrics = {};
                const lines = metricsText.split('\\n');
                
                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i].trim();
                    
                    // Skip comments and empty lines
                    if (line === '' || line.startsWith('#')) continue;
                    
                    // Parse the metric
                    const parts = line.split(' ');
                    if (parts.length >= 2) {
                        const metricName = parts[0];
                        const metricValue = parseFloat(parts[1]);
                        
                        // Parse labels if any
                        let labels = {};
                        if (metricName.includes('{')) {
                            const nameAndLabels = metricName.split('{');
                            const baseName = nameAndLabels[0];
                            const labelsStr = nameAndLabels[1].replace('}', '');
                            
                            const labelParts = labelsStr.split(',');
                            labelParts.forEach(part => {
                                const labelPair = part.split('=');
                                if (labelPair.length === 2) {
                                    const labelName = labelPair[0];
                                    const labelValue = labelPair[1].replace(/"/g, '');
                                    labels[labelName] = labelValue;
                                }
                            });
                            
                            // Add to metrics object
                            if (!metrics[baseName]) metrics[baseName] = [];
                            metrics[baseName].push({
                                labels: labels,
                                value: metricValue
                            });
                        } else {
                            // Simple metric without labels
                            metrics[metricName] = metricValue;
                        }
                    }
                }
                
                return metrics;
            }
            
            // Function to fetch and display metrics
            async function fetchMetrics() {
                try {
                    const response = await fetch('/admin/metrics');
                    const metricsText = await response.text();
                    const metrics = parsePrometheusMetrics(metricsText);
                    
                    console.log('Parsed metrics:', metrics);
                    
                    // Update timestamp
                    document.getElementById('update-time').textContent = new Date().toLocaleString();
                    
                    // Update sentiment predictions table
                    const sentimentTable = document.getElementById('sentiment-table');
                    sentimentTable.innerHTML = '<tr><th>Metric</th><th>Value</th></tr>';
                    
                    const predictionMetrics = metrics['sentiment_predictions_total'] || [];
                    let totalPredictions = 0;
                    
                    predictionMetrics.forEach(metric => {
                        const sentiment = metric.labels.sentiment;
                        const count = metric.value;
                        totalPredictions += count;
                        
                        const row = sentimentTable.insertRow();
                        const cell1 = row.insertCell(0);
                        const cell2 = row.insertCell(1);
                        
                        cell1.textContent = `${sentiment.charAt(0).toUpperCase() + sentiment.slice(1)} predictions`;
                        cell2.textContent = count.toFixed(0);
                    });
                    
                    const totalRow = sentimentTable.insertRow();
                    const totalCell1 = totalRow.insertCell(0);
                    const totalCell2 = totalRow.insertCell(1);
                    totalCell1.textContent = 'Total predictions';
                    totalCell1.style.fontWeight = 'bold';
                    totalCell2.textContent = totalPredictions.toFixed(0);
                    totalCell2.style.fontWeight = 'bold';
                    
                    // Update performance table
                    const performanceTable = document.getElementById('performance-table');
                    performanceTable.innerHTML = '<tr><th>Metric</th><th>Value</th></tr>';
                    
                    // Calculate average latency
                    let avgLatency = 'N/A';
                    if (metrics['sentiment_prediction_latency_seconds_sum'] && 
                        metrics['sentiment_prediction_latency_seconds_count']) {
                        const sum = metrics['sentiment_prediction_latency_seconds_sum'];
                        const count = metrics['sentiment_prediction_latency_seconds_count'];
                        if (count > 0) {
                            avgLatency = ((sum / count) * 1000).toFixed(2) + ' ms';
                        }
                    }
                    
                    const latencyRow = performanceTable.insertRow();
                    const latencyCell1 = latencyRow.insertCell(0);
                    const latencyCell2 = latencyRow.insertCell(1);
                    latencyCell1.textContent = 'Average prediction latency';
                    latencyCell2.textContent = avgLatency;
                    
                    // Update text metrics table
                    const textTable = document.getElementById('text-table');
                    textTable.innerHTML = '<tr><th>Metric</th><th>Value</th></tr>';
                    
                    // Calculate average text length
                    let avgTextLength = 'N/A';
                    if (metrics['sentiment_text_length_sum'] && 
                        metrics['sentiment_text_length_count']) {
                        const sum = metrics['sentiment_text_length_sum'];
                        const count = metrics['sentiment_text_length_count'];
                        if (count > 0) {
                            avgTextLength = (sum / count).toFixed(1) + ' characters';
                        }
                    }
                    
                    const textLengthRow = textTable.insertRow();
                    const textLengthCell1 = textLengthRow.insertCell(0);
                    const textLengthCell2 = textLengthRow.insertCell(1);
                    textLengthCell1.textContent = 'Average text length';
                    textLengthCell2.textContent = avgTextLength;
                    
                    // Calculate average confidence
                    let avgConfidence = 'N/A';
                    if (metrics['sentiment_confidence_sum'] && 
                        metrics['sentiment_confidence_count']) {
                        const sum = metrics['sentiment_confidence_sum'];
                        const count = metrics['sentiment_confidence_count'];
                        if (count > 0) {
                            avgConfidence = (sum / count).toFixed(2);
                        }
                    }
                    
                    const confidenceRow = textTable.insertRow();
                    const confidenceCell1 = confidenceRow.insertCell(0);
                    const confidenceCell2 = confidenceRow.insertCell(1);
                    confidenceCell1.textContent = 'Average confidence';
                    confidenceCell2.textContent = avgConfidence;
                    
                } catch (error) {
                    console.error('Error fetching metrics:', error);
                    alert('Error fetching metrics: ' + error.message);
                }
            }
            
            // Fetch metrics on page load
            document.addEventListener('DOMContentLoaded', function() {
                fetchMetrics();
                // Auto-refresh every 30 seconds
                setInterval(fetchMetrics, 30000);
            });
        </script>
    </body>
    </html>
    """
    return html_content