"""Local development entry point. Production uses wsgi.py via gunicorn."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app import create_app
from app.core.config import get_config

app = create_app()

if __name__ == "__main__":
    config = get_config()
    app.run(host="0.0.0.0", port=config.port, debug=config.debug)
