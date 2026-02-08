"""
Django settings for job_verifier_project project.
"""

from pathlib import Path
import os  # used for env variables (Render safe)

BASE_DIR = Path(__file__).resolve().parent.parent


# =========================
# SECURITY
# =========================

# ❌ Hardcoded secret key (unsafe for production)
# SECRET_KEY = 'django-insecure-qsr270=nu&txu1+d0^bx(mtlv90)dm4^bkiuf)p5*myw3a5f$3'

# ✅ Use env variable (Render)
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "unsafe-dev-key")

# ❌ DEBUG should NOT be True on Render (causes 500 + memory issues)
# DEBUG = True

# ✅ Production safe
DEBUG = False


ALLOWED_HOSTS = [
    "localhost",
    "127.0.0.1",
    "fake-job-detection-and-resume-analyzer.onrender.com",
    ".onrender.com",
]


# =========================
# APPLICATIONS
# =========================

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # ✅ Correct – loads app config
    'main_app.apps.MainAppConfig',

    # ❌ Do NOT add 'main_app' again
    # 'main_app',
]


MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # required for frontend/API
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'job_verifier_project.urls'


# =========================
# TEMPLATES
# =========================

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],  # HTML files
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]


# =========================
# SERVER
# =========================

WSGI_APPLICATION = 'job_verifier_project.wsgi.application'

# ❌ ASGI not required (Gunicorn uses WSGI)
# ASGI_APPLICATION = 'job_verifier_project.asgi.application'


# =========================
# DATABASE
# =========================

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# =========================
# AUTH
# =========================

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]


# =========================
# I18N
# =========================

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True


# =========================
# STATIC FILES
# =========================

STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / 'static']


DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# =========================
# ML PATHS (kept but NOT auto-used)
# =========================

# ⚠️ These paths are fine, but ML loading must be lazy
MODEL_SAVE_PATH = BASE_DIR / 'Models/hybrid_model_glove_v2.pth'
VOCAB_SAVE_PATH = BASE_DIR / 'Models/vocab_glove_v2.pth'
COLS_SAVE_PATH = BASE_DIR / 'Models/train_cols_glove_v2.pkl'


# =========================
# CORS
# =========================

CORS_ALLOW_ALL_ORIGINS = True
