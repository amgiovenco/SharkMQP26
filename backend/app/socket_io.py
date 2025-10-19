from socketio.async_server import AsyncServer

sio = AsyncServer(async_mode='asgi', cors_allowed_origins='*')