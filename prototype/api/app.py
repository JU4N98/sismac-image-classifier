import logging
import os
from db import db
from endpoints import init_routes
from flask import Flask
from flask_cors import CORS


# LOGGING CONFIGURATION
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO
)
LOG = logging.getLogger(__name__)

# FLASK APPLICATION
DB_USERNAME:str = os.getenv("DB_USERNAME")
DB_PASSWORD:str = os.getenv("DB_PASSWORD")
DB_HOST:str = os.getenv("DB_HOST")
DB_PORT:str = os.getenv("DB_PORT")
DB_NAME:str = os.getenv("DB_NAME")
app = Flask(__name__)
CORS(app)
app.config["SQLALCHEMY_DATABASE_URI"] = f"postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

db.init_app(app)
with app.app_context():
    db.create_all()
init_routes(app)
