from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from app.controllers.pagination import StandardResultsPagination
from app.serializers.document_serializers import (
    DocumentDetailSerializer,
    DocumentSerializer,
    DocumentUploadRequestSerializer,
    DocumentUploadResultSerializer,
)
from app.services import container


class DocumentListCreateController(APIView):
    """GET: list the user's documents. POST: upload + index one or more PDF/TXT files."""
    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(tags=['Documents'], parameters=[
        OpenApiParameter('page', int), OpenApiParameter('page_size', int)],
        responses={200: DocumentSerializer(many=True)})
    def get(self, request):
        documents = container.build_document_service().list_documents(request.user)
        paginator = StandardResultsPagination()
        page = paginator.paginate_queryset(documents, request, view=self)
        return paginator.get_paginated_response(DocumentSerializer(page, many=True).data)

    @extend_schema(tags=['Documents'], request={'multipart/form-data': DocumentUploadRequestSerializer},
                   responses={201: DocumentUploadResultSerializer(many=True)})
    def post(self, request):
        files = request.FILES.getlist('files') or request.FILES.getlist('file')
        results = container.build_document_service().upload_documents(
            user=request.user, files=files, title=request.data.get('title'))
        return Response({'results': DocumentUploadResultSerializer(results, many=True).data},
                        status=status.HTTP_201_CREATED)


class DocumentDetailController(APIView):
    @extend_schema(tags=['Documents'], responses={200: DocumentDetailSerializer})
    def get(self, request, pk):
        document = container.build_document_service().get_document(request.user, pk)
        return Response(DocumentDetailSerializer(document).data)

    @extend_schema(tags=['Documents'], responses={204: None})
    def delete(self, request, pk):
        container.build_document_service().delete_document(request.user, pk)
        return Response(status=status.HTTP_204_NO_CONTENT)


class DocumentReingestController(APIView):
    @extend_schema(tags=['Documents'], request=None, responses={200: dict})
    def post(self, request, pk):
        result = container.build_document_service().reingest_document(request.user, pk)
        return Response(result, status=status.HTTP_200_OK)
