from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from app.serializers.auth_serializers import (
    AuthTokenResponseSerializer,
    LoginSerializer,
    LogoutSerializer,
    MeResponseSerializer,
    RegisterSerializer,
)
from app.services import container


def _token_payload(user, raw_token, session):
    return {
        'token': raw_token,
        'token_type': 'Bearer',
        'expires_at': session.expires_at,
        'user': {'id': user.pk, 'username': user.username, 'email': user.email},
    }


class RegisterController(APIView):

    @extend_schema(tags=['Auth'], request=RegisterSerializer, responses={201: AuthTokenResponseSerializer})
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        user, token, session = container.build_auth_service().register_user(
            username=data['username'], password=data['password'], email=data.get('email'),
            user_agent=request.headers.get('User-Agent', ''))
        return Response(_token_payload(user, token, session), status=status.HTTP_201_CREATED)


class LoginController(APIView):

    @extend_schema(tags=['Auth'], request=LoginSerializer, responses={200: AuthTokenResponseSerializer})
    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user, token, session = container.build_auth_service().login_user(
            username=serializer.validated_data['username'], password=serializer.validated_data['password'],
            user_agent=request.headers.get('User-Agent', ''))
        return Response(_token_payload(user, token, session), status=status.HTTP_200_OK)


class LogoutController(APIView):
    @extend_schema(tags=['Auth'], request=LogoutSerializer, responses={204: None})
    def post(self, request):
        serializer = LogoutSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        container.build_auth_service().logout(
            session=getattr(request._request, 'user_session', None),
            all_sessions=serializer.validated_data['all_sessions'])
        return Response(status=status.HTTP_204_NO_CONTENT)


class MeController(APIView):
    @extend_schema(tags=['Auth'], responses={200: MeResponseSerializer})
    def get(self, request):
        session = getattr(request._request, 'user_session', None)
        return Response({
            'id': request.user.pk, 'username': request.user.username, 'email': request.user.email,
            'session_expires_at': session.expires_at if session else None,
        })
