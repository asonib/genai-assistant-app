from django.conf import settings
from django.db import connection
from drf_spectacular.utils import extend_schema
from rest_framework.response import Response
from rest_framework.views import APIView


class HealthController(APIView):
    """GET /api/health — public liveness/readiness probe (no secrets exposed)."""

    @extend_schema(tags=['Health'], responses={200: dict})
    def get(self, request):
        try:
            with connection.cursor() as cursor:
                cursor.execute('SELECT 1')
            db_ok = True
        except Exception:
            db_ok = False
        return Response({
            'status': 'ok' if db_ok else 'degraded',
            'database': db_ok,
            'llm_configured': bool(settings.ANTHROPIC_API_KEY),
            'llm_model': settings.ANTHROPIC_MODEL,
            'mcp_enabled': settings.MCP_ENABLED,
            'mcp_transport': settings.MCP_TRANSPORT if settings.MCP_ENABLED else 'disabled',
        }, status=200 if db_ok else 503)
