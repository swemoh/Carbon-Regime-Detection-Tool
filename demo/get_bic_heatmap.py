import seaborn as sns
import pandas as pd
import numpy as np
import datetime

import matplotlib
matplotlib.use('Agg') ## Adding to avoid assertion failed error. It would shut down the server.
import matplotlib.pyplot as plt

# for static images
import io
import base64


from build_cluster_map import calculate_BIC