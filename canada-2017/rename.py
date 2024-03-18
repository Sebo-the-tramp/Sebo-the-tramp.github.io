#rename every photo with integer from 1 to n

import os
import sys

def rename(path):
    i = 1
    for filename in os.listdir(path):
        if filename.endswith(".JPG"):
            os.rename(os.path.join(path, filename), os.path.join(path, str(i) + ".JPG"))
            i += 1
    
if __name__ == "__main__":
    rename("./")