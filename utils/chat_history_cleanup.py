try: 
    from chat_history_management import chat_cleanup_operation
except:
    pass
from time import sleep
import schedule
from pydantic import BaseModel

class DatabaseUpdate(BaseModel):
    update_period: int = 60*24

# define the settings model
settings_schedule = DatabaseUpdate()

# Schedule the task to run every 2 minutes
schedule.every(settings_schedule.update_period).minutes.do(chat_cleanup_operation)

while True:
    schedule.run_pending()
    sleep(1)