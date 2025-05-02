from app.monitoring.metrics import start_metrics_server
from app.utils.config import get_settings

settings = get_settings()
start_metrics_server(port = settings.METRICS_PORT)