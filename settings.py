from pathlib import Path

PATH_TO_HERE = Path(__file__).parent

DB_URI = f"sqlite:///{PATH_TO_HERE}/app.db"

SECRET_KEY = "IkjPce9DTy12wnRr30WOmT105vkxg8-ZTWkU28Y9fjM"
MODEL_REASONING = "o3-mini"
MODEL_DEFAULT = "gpt-4o-mini"
