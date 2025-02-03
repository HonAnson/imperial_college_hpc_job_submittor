import os
import time
from datetime import datetime

for i in range(4):
    # Submit the PBS script
    os.system(f"qdel {i+156443}")
    time.sleep(1)



