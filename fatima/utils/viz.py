import glob
import os
import subprocess
import time

import matplotlib.pyplot as plt
import numpy
import torch


def viz(
    batch: torch.Tensor,
    episodes=1000,
    video=True,
    folder='output',
) -> None:
  ## Visualize GoodAI Breakout Dataset
  fig = plt.figure(1)
  ax = fig.add_subplot(111)
  ax.set_title("Breakout")

  im = ax.imshow(numpy.zeros((84, 84, 4)))  # Blank starting image
  fig.show()
  im.axes.figure.canvas.draw()

  tstart = time.time()
  rewards = 0
  for episode in range(episodes):
    image = batch.states[episode].permute(1, 2, 0)
    rewards += batch.rewards[episode].detach().cpu().numpy()
    ax.set_title(str(f"episode: {episode} | reward: {rewards}"))
    im.set_data(image)
    im.axes.figure.canvas.draw()
    ax.figure.savefig(folder + "/img%02d.png" % episode)

  if video:
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', f'{folder}/img%02d.png', '-r', '30',
        '-pix_fmt', 'yuv420p', f'{folder}/video_name.mp4'
    ])
    for file_name in glob.glob(f"{folder}/*.png"):
      os.remove(file_name)

  print('FPS:', 100 / (time.time() - tstart))
