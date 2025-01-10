from redis import Redis
from typing import Any, Optional

class ApplicationStateManager:
    def __init__(self):
        self.redis = Redis(
            host='5.78.113.143',  # Your server IP
            port=6379,
            #password='CustomAI1234',  # The password you set
            db=0,
            socket_timeout=5
        )
        
    def get_state(self, key: str, default: Any = None) -> Any:
        try:
            value = self.redis.get(key)
            return value if value is not None else default
        except Exception as e:
            print(f"Redis error: {e}")
            return default
        
    def set_state(self, key: str, value: Any) -> bool:
        try:
            return self.redis.set(key, value)
        except Exception as e:
            print(f"Redis error: {e}")
            return False