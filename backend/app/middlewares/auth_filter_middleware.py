from django.http import JsonResponse
from django.utils import timezone

from app.config.app_config import AppConfigConstants
from app.models.user_session import UserSession


def _unauthorized(detail, request):
    body = {'detail': detail, 'code': 'not_authenticated'}
    request_id = getattr(request, 'request_id', None)
    if request_id:
        body['request_id'] = request_id
    return JsonResponse(body, status=401)


class TokenAuthFilterMiddleware:
    """
    Filter layer: authenticates every non-public request *before* it reaches a
    controller and attaches `request.user` / `request.user_session`.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        path = request.path_info
        if any(path.startswith(prefix) for prefix in AppConfigConstants.PUBLIC_URL_PREFIXES):
            return self.get_response(request)

        auth_header = request.headers.get('Authorization', '')
        if not auth_header:
            return _unauthorized('Authentication credentials were not provided.', request)

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() not in ('bearer', 'token'):
            return _unauthorized('Invalid token header format. Use: Bearer <token> or Token <token>.', request)

        session = UserSession.find_by_raw_token(parts[1])
        if session is None:
            return _unauthorized('Invalid authentication token.', request)
        if session.is_expired():
            session.delete()
            return _unauthorized('Token has expired. Please log in again.', request)
        if not session.user.is_active:
            return _unauthorized('User account is disabled.', request)

        # Cheap "last seen" bookkeeping, at most once a minute per session.
        now = timezone.now()
        if (now - session.last_seen_at).total_seconds() > 60:
            UserSession.objects.filter(pk=session.pk).update(last_seen_at=now)

        request.user = session.user
        request.user_session = session
        return self.get_response(request)
