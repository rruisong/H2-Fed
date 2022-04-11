import matplotlib.pyplot as plt
from post_processing.recorder import Recorder
import os

recorder = Recorder()

res_dir = 'results/Non-IID_Agent_Layer'
# res_dir = 'results/Non-IID_RSU_Layer'

res_files = [f for f in os.listdir(res_dir)]
for f in res_files:
    recorder.load(os.path.join(res_dir, f), label=f)
recorder.plot()
plt.show()