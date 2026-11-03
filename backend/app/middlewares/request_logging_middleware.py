import logging
import re
import time
import uuid

from app.config.app_config import AppConfigConstants

logger = logging.getLogger('app.request')

_SAFE_ID = re.compile(r'^[A-Za-z0-9._-]{1,64}$')


class RequestLoggingMiddleware:
    """
    Outermost app middleware: assigns a request id (honours a sane incoming
    X-Request-ID, e.g. from the Streamlit client), times the request and emits
    one structured access-log line per request. Never logs bodies or tokens.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        incoming = request.headers.get(AppConfigConstants.REQUEST_ID_HEADER, '')
        request.request_id = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex
        start = time.perf_counter()
        response = self.get_response(request)
        duration_ms = int((time.perf_counter() - start) * 1000)

        user = getattr(request, 'user', None)
        user_id = user.pk if getattr(user, 'is_authenticated', False) else None
        level = logging.ERROR if response.status_code >= 500 else (
            logging.WARNING if response.status_code >= 400 else logging.INFO)
        logger.log(
            level,
            'method=%s path=%s status=%s duration_ms=%s user_id=%s request_id=%s',
            request.method, request.path_info, response.status_code, duration_ms, user_id, request.request_id,
        )
        response[AppConfigConstants.REQUEST_ID_HEADER] = request.request_id
        return response
