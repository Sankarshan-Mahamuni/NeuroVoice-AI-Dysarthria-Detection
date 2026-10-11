"""Entry point for the AI-Based Dysarthria Detection web application."""
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("dysarthria_app")

from app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting NeuroVoice AI Dysarthria Detection Server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
