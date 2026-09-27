from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.contracts import ErrorResponse
from app.request_id import generate_request_id


REQUEST_ID_HEADER = "X-Request-ID"


def get_request_id(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)

    if request_id:
        return request_id

    request_id = request.headers.get(REQUEST_ID_HEADER) or generate_request_id()
    request.state.request_id = request_id
    return request_id


def error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    request_id = get_request_id(request)

    payload = ErrorResponse(
        code=code,
        message=message,
        request_id=request_id,
    )

    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(),
        headers={REQUEST_ID_HEADER: request_id},
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return error_response(
        request,
        status_code=422,
        code="VALIDATION_ERROR",
        message="Request validation failed.",
    )


async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else "HTTP request failed."

    return error_response(
        request,
        status_code=exc.status_code,
        code=f"HTTP_{exc.status_code}",
        message=message,
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    return error_response(
        request,
        status_code=500,
        code="INTERNAL_ERROR",
        message="Internal server error.",
    )
