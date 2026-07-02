import os
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
SECRET_KEY = "repro-harness-not-secret"
DEBUG = True
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = ["pipeline"]
MIDDLEWARE = []
ROOT_URLCONF = "config.urls"
TEMPLATES = []
WSGI_APPLICATION = None
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
USE_TZ = True

if os.environ.get("POSTGRES_HOST"):
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.environ.get("POSTGRES_DB", "repro"),
            "USER": os.environ.get("POSTGRES_USER", "repro"),
            "PASSWORD": os.environ.get("POSTGRES_PASSWORD", "repro"),
            "HOST": os.environ.get("POSTGRES_HOST", "db"),
            "PORT": os.environ.get("POSTGRES_PORT", "5432"),
        }
    }
else:  # local fallback for quick smoke tests without Postgres
    DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3",
                             "NAME": BASE_DIR / "repro.sqlite3"}}
