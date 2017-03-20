
# coding: utf-8

# In[ ]:

from PIL import Image
import numpy as np
import sys
im = Image.open(sys.argv[1]).rotate(180)
im.save('ans2.png')

