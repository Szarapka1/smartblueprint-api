# app/dependencies.py
from fastapi import Request
from app.services.ai_service import AIService
from app.services.storage_service import StorageService
from app.services.pdf_service import PDFService
from app.services.session_service import SessionService

def get_storage_service(request: Request) -> StorageService:
    return request.app.state.storage_service

def get_pdf_service(request: Request) -> PDFService:
    return request.app.state.pdf_service

def get_ai_service(request: Request) -> AIService:
    return request.app.state.ai_service

def get_session_service(request: Request) -> SessionService:
    return request.app.state.session_service