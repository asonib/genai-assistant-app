import logging

from rest_framework import status
from rest_framework.exceptions import APIException
from rest_framework.views import exception_handler

logger = logging.getLogger(__name__)


class PayloadTooLargeException(APIException):
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    default_detail = 'File size exceeds maximum allowed limit.'
    default_code = 'file_too_large'


class UnsupportedFileTypeException(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = 'File type is not permitted.'
    default_code = 'invalid_file_type'


class ResourceNotFoundException(APIException):
    status_code = status.HTTP_404_NOT_FOUND
    default_detail = 'Requested resource not found.'
    default_code = 'not_found'


class EmptyDocumentException(APIException):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    default_detail = 'Document has no extractable text content to ingest.'
    default_code = 'empty_document'


class TextExtractionException(APIException):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    default_detail = 'Could not extract text from the uploaded file.'
    default_code = 'extraction_failed'


class LLMNotConfiguredException(APIException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_detail = 'LLM provider is not configured. Set ANTHROPIC_API_KEY in the environment.'
    default_code = 'llm_not_configured'


class LLMServiceException(APIException):
    status_code = status.HTTP_502_BAD_GATEWAY
    default_detail = 'The upstream LLM provider failed to produce a response.'
    default_code = 'llm_upstream_error'


class EmbeddingServiceException(APIException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_detail = 'Embedding model is unavailable.'
    default_code = 'embedding_unavailable'


class VectorStoreException(APIException):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_detail = 'Vector store operation failed.'
    default_code = 'vector_store_error'


class AgentExecutionException(APIException):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_detail = 'The assistant agent failed to complete the request.'
    default_code = 'agent_error'


class ExternalToolException(APIException):
    """Raised by the MCP client; the agent converts it into an is_error tool result."""
    status_code = status.HTTP_502_BAD_GATEWAY
    default_detail = 'External tool call failed.'
    default_code = 'external_tool_error'


def custom_exception_handler(exc, context):
    """
    DRF handler + a stable error envelope:
      {"detail": ..., "code": ..., "request_id": ...}
    Validation errors keep DRF's field map under "errors".
    """
    response = exception_handler(exc, context)
    if response is None:
        return None
    request = context.get('request')
    request_id = getattr(getattr(request, '_request', request), 'request_id', None)
    data = response.data
    if isinstance(data, dict) and 'detail' in data:
        code = getattr(data['detail'], 'code', None) or getattr(exc, 'default_code', 'error')
        body = {'detail': str(data['detail']), 'code': code}
    else:
        body = {'detail': 'Validation failed.', 'code': 'invalid', 'errors': data}
    if request_id:
        body['request_id'] = request_id
    if response.status_code >= 500:
        logger.error('API error %s: %s', response.status_code, body['detail'])
    response.data = body
    return response
