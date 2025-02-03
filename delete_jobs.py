import os
import time
from datetime import datetime

for i in range(15):
    # Submit the PBS script
    os.system(f"qdel {i+156411}")
    time.sleep(1)



