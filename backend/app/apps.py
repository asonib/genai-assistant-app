from django.apps import AppConfig

class AppConfigCore(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'

    def ready(self):
        # Register drf-spectacular extension for the custom auth bridge.
        from app.middlewares import openapi_auth_extension  # noqa: F401
