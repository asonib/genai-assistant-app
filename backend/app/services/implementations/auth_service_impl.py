from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError as DjangoValidationError
from django.db import transaction
from rest_framework.exceptions import AuthenticationFailed, ValidationError

from app.models.user_session import UserSession
from app.services.interfaces.auth_service_interface import IAuthService


class AuthServiceImpl(IAuthService):
    def register_user(self, username, password, email=None, user_agent=''):
        if User.objects.filter(username__iexact=username).exists():
            raise ValidationError({'username': ['Username already taken.']})
        candidate = User(username=username, email=email or '')
        try:
            validate_password(password, user=candidate)
        except DjangoValidationError as exc:
            raise ValidationError({'password': list(exc.messages)})
        with transaction.atomic():
            user = User.objects.create_user(username=username, password=password, email=email or '')
            session, raw_token = UserSession.create_session(user, user_agent)
        return user, raw_token, session

    def login_user(self, username, password, user_agent=''):
        user = authenticate(username=username, password=password)
        if not user:
            raise AuthenticationFailed('Invalid username or password.')
        session, raw_token = UserSession.create_session(user, user_agent)
        return user, raw_token, session

    def logout(self, session, all_sessions=False) -> int:
        if session is None:
            return 0
        if all_sessions:
            deleted, _ = UserSession.objects.filter(user_id=session.user_id).delete()
        else:
            deleted, _ = UserSession.objects.filter(pk=session.pk).delete()
        return deleted
