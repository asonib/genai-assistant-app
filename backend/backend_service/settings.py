"""
Django settings for the GenAI Assistant backend (Assignment 3B).

Everything environment-specific is read from <repo-root>/.env (never committed).
Environment-agnostic constants live in app/config/app_config.py.
"""
import os
import shlex
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent          # backend/
REPO_ROOT = BASE_DIR.parent                                # genai-assistant-app/

load_dotenv(REPO_ROOT / '.env')
load_dotenv(BASE_DIR / '.env')  # optional backend-local override


def _env_bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in ('1', 'true', 'yes', 'on')


DEBUG = _env_bool('DEBUG', True)
SECRET_KEY = os.getenv('SECRET_KEY', '')
if not SECRET_KEY:
    if not DEBUG:
        raise RuntimeError('SECRET_KEY must be set when DEBUG=False.')
    SECRET_KEY = 'django-insecure-dev-only-genai-assistant-key'

ALLOWED_HOSTS = [h.strip() for h in os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',') if h.strip()]
TESTING = len(sys.argv) > 1 and sys.argv[1] == 'test'
if TESTING:
    ALLOWED_HOSTS.append('testserver')

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'drf_spectacular',
    'app',
]

MIDDLEWARE = [
    'app.middlewares.request_logging_middleware.RequestLoggingMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'app.middlewares.auth_filter_middleware.TokenAuthFilterMiddleware',
]

ROOT_URLCONF = 'backend_service.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'backend_service.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': Path(os.getenv('SQLITE_PATH', BASE_DIR / 'db.sqlite3')),
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', 'OPTIONS': {'min_length': 8}},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Uploads: keep Django's in-memory threshold aligned with our per-file cap.
DATA_UPLOAD_MAX_MEMORY_SIZE = 12 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 12 * 1024 * 1024

if not DEBUG:
    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = 'DENY'
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True

# ---------------------------------------------------------------------------
# AI layer: LLM, embeddings, vector store
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-5')
VECTOR_STORE_DIR = Path(os.getenv('VECTOR_STORE_DIR', BASE_DIR / 'vector_store'))
EMBEDDING_CACHE_DIR = os.getenv('EMBEDDING_CACHE_DIR') or None

# ---------------------------------------------------------------------------
# External tools: custom MCP server (mcp_server/server.py)
#   MCP_TRANSPORT=stdio -> the backend spawns the server as a subprocess
#   MCP_TRANSPORT=http  -> connect to a running server at MCP_SERVER_URL
#   MCP_ENABLED=False   -> agent runs with document search only
# ---------------------------------------------------------------------------
MCP_ENABLED = _env_bool('MCP_ENABLED', True)
MCP_TRANSPORT = os.getenv('MCP_TRANSPORT', 'stdio').strip().lower()
MCP_SERVER_COMMAND = os.getenv('MCP_SERVER_COMMAND', sys.executable)
MCP_SERVER_ARGS = shlex.split(os.getenv('MCP_SERVER_ARGS', ''), posix=os.name != 'nt') or [
    str(REPO_ROOT / 'mcp_server' / 'server.py')
]
MCP_SERVER_URL = os.getenv('MCP_SERVER_URL', 'http://127.0.0.1:8765/mcp')

# ---------------------------------------------------------------------------
# Logging: console + rotating file (logs/backend.log)
# ---------------------------------------------------------------------------
LOG_DIR = Path(os.getenv('LOG_DIR', BASE_DIR / 'logs'))
LOG_LEVEL = os.getenv('APP_LOG_LEVEL', 'INFO')
_handlers = {'console': {'class': 'logging.StreamHandler', 'formatter': 'standard'}}
if not TESTING:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _handlers['file'] = {
        'class': 'logging.handlers.RotatingFileHandler',
        'filename': str(LOG_DIR / 'backend.log'),
        'maxBytes': 5 * 1024 * 1024,
        'backupCount': 5,
        'encoding': 'utf-8',
        'formatter': 'standard',
    }

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {'format': '%(asctime)s %(levelname)s %(name)s %(message)s'},
    },
    'handlers': _handlers,
    'loggers': {
        'app': {
            'handlers': list(_handlers),
            'level': 'CRITICAL' if TESTING else LOG_LEVEL,
            'propagate': False,
        },
    },
}

REST_FRAMEWORK = {
    # Bridge the user authenticated by TokenAuthFilterMiddleware into DRF.
    # No SessionAuthentication -> no CSRF failures on token-authenticated POSTs.
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'app.middlewares.drf_passthrough_auth.FilterPassthroughAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [],
    'EXCEPTION_HANDLER': 'app.exceptions.custom_exceptions.custom_exception_handler',
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema',
}

SPECTACULAR_SETTINGS = {
    'TITLE': 'GenAI Assistant Backend API',
    'DESCRIPTION': (
        'Django/DRF backend for the Streamlit GenAI Assistant: token auth, PDF/TXT ingestion into FAISS, '
        'and a LangGraph agent that combines RAG with external MCP tools (weather, FX) and conversation memory.'
    ),
    'VERSION': '3.1.0',
    'SERVE_INCLUDE_SCHEMA': False,
    'SECURITY': [{'TokenAuth': []}],
}
