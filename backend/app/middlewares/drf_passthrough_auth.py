from rest_framework.authentication import BaseAuthentication


class FilterPassthroughAuthentication(BaseAuthentication):
    """
    Bridges the user attached to the request by TokenAuthFilterMiddleware into
    DRF's request.user (otherwise DRF would reset it to AnonymousUser).
    """

    def authenticate(self, request):
        django_user = getattr(request._request, 'user', None)
        if django_user and django_user.is_authenticated:
            return (django_user, getattr(request._request, 'user_session', None))
        return None

    def authenticate_header(self, request):
        # Makes DRF answer AuthenticationFailed with 401 (+ WWW-Authenticate) instead of 403.
        return 'Bearer'
