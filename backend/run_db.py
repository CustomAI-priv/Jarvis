try: 
    from setup import DataLoader
except:
    pass
from time import sleep
import schedule
from pydantic import BaseModel

class DatabaseUpdate(BaseModel):
    update_period: int = 2

# run the database updating operatinon 
def run_db():
    data_loader = DataLoader()
    data_loader.facade()
    print('\n\n')

# define the settings model
settings_schedule = DatabaseUpdate()

# Schedule the task to run every 2 minutes
schedule.every(settings_schedule.update_period).minutes.do(run_db)

while True:
    schedule.run_pending()
    sleep(1)

