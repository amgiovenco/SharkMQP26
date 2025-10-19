import json
import asyncio
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import from_url
from .db import Base, engine
from .auth import router as auth_router
from .cases import router as cases_router
from .jobs import router as jobs_router
from .settings import settings
from contextlib import asynccontextmanager
from socketio.asgi import ASGIApp
from .socket_io import sio
from .logger import get_logger

logger = get_logger(__name__)

redis_listener_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_listener_task
    Base.metadata.create_all(bind=engine)
    
    # Start Redis listener
    redis_listener_task = asyncio.create_task(listen_to_redis_updates())
    logger.info("Started Redis listener task")
    
    yield
    
    # Cleanup
    if redis_listener_task:
        redis_listener_task.cancel()
        try:
            await redis_listener_task
        except asyncio.CancelledError:
            logger.info("Redis listener task cancelled")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.allowed_origins] if settings.allowed_origins != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up API routers
api_router = APIRouter()
api_router.include_router(auth_router, prefix="/auth")
api_router.include_router(jobs_router, prefix="/jobs")
api_router.include_router(cases_router, prefix="/cases")

app.include_router(api_router, prefix="/api")

# Socket.IO event handlers
@sio.event
async def connect(sid, environ, auth):
    """Handle socket connection with JWT auth"""
    token = auth.get('token') if auth else None
    if not token:
        logger.warning(f"Socket client {sid} attempted connection without token")
        return False
    logger.info(f"Socket client {sid} connected")
    return True

@sio.event
async def disconnect(sid):
    logger.info(f"Socket client {sid} disconnected")

# Redis listener for job status updates
async def listen_to_redis_updates():
    """Listen for job status updates from worker via Redis pub/sub"""
    r = None
    pubsub = None

    # Subscribe to Redis channel
    try:
        r = await from_url(settings.redis_url, decode_responses=True)
        pubsub = r.pubsub()
        await pubsub.subscribe('job_status_updates')
        logger.info("Redis job status listener started and subscribed")
        
        # Listen for messages indefinitely
        async for message in pubsub.listen():
            if message['type'] == 'message':

                # Broadcast to all connected Socket.IO clients
                try:
                    data = json.loads(message['data'])
                    logger.info(f"Broadcasting job status: {data.get('job_id')} -> {data.get('status')}")
                    await sio.emit('job_status', data)
                    logger.info(f"Emitted job_status event to all connected clients")

                except Exception as e:
                    logger.error(f"Error broadcasting job status: {e}", exc_info=True)
    
    except asyncio.CancelledError:
        logger.info("Redis listener cancelled")
        raise
    
    except Exception as e:
        logger.error(f"Redis listener error: {e}", exc_info=True)
    
    # Cleanup on exit
    finally:
        if pubsub:
            await pubsub.unsubscribe('job_status_updates')
            await pubsub.close()
        if r:
            await r.close()

# Mount Socket.IO
sio_asgi_app = ASGIApp(
    socketio_server=sio,
    other_asgi_app=app,
    socketio_path='socket.io'
)