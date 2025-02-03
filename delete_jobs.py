import os
import time
from datetime import datetime

for i in range(12):
    # Submit the PBS script
    os.system(f"qdel {i+161822}")
    time.sleep(1)



