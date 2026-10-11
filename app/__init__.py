"""Flask application factory."""
import logging

from flask import Flask

from config import Config
from app.services.model_loader import ModelManager

logger = logging.getLogger("dysarthria_app")


def create_app(config_class=Config):
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(config_class)
    app.config["BASE_DIR"] = str(Config.ROOT_DIR)
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_REQUEST_SIZE_BYTES

    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)

    logger.info("Loading trained model for Flask application.")
    model_mgr = ModelManager.get_instance()
    logger.info("Model ready: %s", model_mgr.config.get("model_name"))
    return app
