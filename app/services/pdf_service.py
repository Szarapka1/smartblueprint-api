@blueprint_router.post("/documents/upload", status_code=201)
async def upload_shared_document(
    request: Request,
    document_id: str,
    file: UploadFile = File(...)
):
    """
    Uploads a PDF blueprint as a shared document that everyone can access.
    Uses a custom document_id instead of random session_id.
    """
    logger = logging.getLogger("blueprint_upload")
    
    try:
        logger.info(f"🚀 Starting upload for document_id: '{document_id}'")
        
        # Validate file type
        if not file.content_type or file.content_type != "application/pdf":
            logger.warning(f"❌ Invalid file type: {file.content_type}")
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Only PDF files are allowed."
            )

        # Validate file size (60MB limit)
        max_size = int(os.getenv("MAX_FILE_SIZE_MB", "60")) * 1024 * 1024
        pdf_bytes = await file.read()
        if len(pdf_bytes) > max_size:
            logger.warning(f"❌ File too large: {len(pdf_bytes)} bytes")
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {max_size // (1024*1024)}MB"
            )

        # Validate and clean the document ID
        clean_document_id = validate_document_id(document_id)
        logger.info(f"✅ Clean document ID: '{clean_document_id}'")
        
        # Check if required services are available
        pdf_service = request.app.state.pdf_service
        storage_service = request.app.state.storage_service
        
        logger.info(f"🔧 Service availability check:")
        logger.info(f"   📁 Storage service: {'✅ Available' if storage_service else '❌ None'}")
        logger.info(f"   📄 PDF service: {'✅ Available' if pdf_service else '❌ None'}")
        
        if not storage_service:
            logger.error("❌ Storage service unavailable")
            raise HTTPException(
                status_code=503,
                detail="Storage service unavailable. Please try again later."
            )
        
        if not pdf_service:
            logger.error("❌ PDF service unavailable")
            raise HTTPException(
                status_code=503,
                detail="PDF processing service unavailable. Please try again later."
            )

        # Check if document already exists
        logger.info(f"🔍 Checking if document '{clean_document_id}' already exists...")
        try:
            existing_context = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}_context.txt"
            )
            logger.info(f"📄 Document '{clean_document_id}' already exists")
            return {
                "document_id": clean_document_id,
                "filename": file.filename,
                "status": "already_exists",
                "message": f"Document '{clean_document_id}' already exists and is ready for use",
                "file_size_mb": round(len(pdf_bytes) / (1024*1024), 2)
            }
        except:
            # Document doesn't exist, proceed with upload
            logger.info(f"📄 Document '{clean_document_id}' is new, proceeding with upload")
            pass

        # Upload original PDF to main container
        logger.info(f"📤 Uploading original PDF to main container...")
        await storage_service.upload_file(
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}.pdf",
            data=pdf_bytes
        )
        logger.info(f"✅ Original PDF uploaded successfully")

        # Process PDF for shared AI and chat use
        logger.info(f"⚙️ Starting PDF processing and caching...")
        try:
            await pdf_service.process_and_cache_pdf(
                session_id=clean_document_id,  # Use document_id as session_id for processing
                pdf_bytes=pdf_bytes,
                storage_service=storage_service
            )
            logger.info(f"✅ PDF processing completed successfully")
        except Exception as processing_error:
            logger.error(f"❌ PDF processing failed: {processing_error}")
            # Clean up the uploaded PDF if processing failed
            try:
                await storage_service.delete_blob(
                    container_name=settings.AZURE_CONTAINER_NAME,
                    blob_name=f"{clean_document_id}.pdf"
                )
                logger.info(f"🧹 Cleaned up original PDF after processing failure")
            except:
                pass
            raise HTTPException(
                status_code=500,
                detail=f"PDF processing failed: {str(processing_error)}"
            )

        # Verify the document is ready for chat
        logger.info(f"🔍 Verifying document is ready for chat...")
        try:
            # Check if all required files exist
            context_exists = await storage_service.blob_exists(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}_context.txt"
            )
            chunks_exists = await storage_service.blob_exists(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}_chunks.json"
            )
            
            logger.info(f"📋 Verification results:")
            logger.info(f"   📄 Context file: {'✅ Exists' if context_exists else '❌ Missing'}")
            logger.info(f"   📑 Chunks file: {'✅ Exists' if chunks_exists else '❌ Missing'}")
            
            if not context_exists or not chunks_exists:
                raise Exception("Required processing files are missing")
                
        except Exception as verification_error:
            logger.error(f"❌ Document verification failed: {verification_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Document processing verification failed: {str(verification_error)}"
            )

        logger.info(f"🎉 Upload and processing completed successfully for '{clean_document_id}'")
        
        return {
            "document_id": clean_document_id,
            "filename": file.filename,
            "status": "processing_complete",
            "message": f"Document '{clean_document_id}' uploaded and ready for collaborative use",
            "file_size_mb": round(len(pdf_bytes) / (1024*1024), 2),
            "ready_for_chat": True
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload failed for '{document_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
