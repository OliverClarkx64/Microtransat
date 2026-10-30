import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import numpy as np

# Load image
img = Image.open("g2.png").convert("RGBA")

fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Initial image
im = ax.imshow(
    img,
    extent=[4, 6, 4, 6]  # x_min, x_max, y_min, y_max
)

def animate(frame):

    angle = frame * 5  # degrees per frame

    # Rotate around image centre
    rotated = img.rotate(
        angle,
##        resample=Image.Resampling.BICUBIC,
##        expand=False
    )

    im.set_data(np.array(rotated))

    return [im]

anim = FuncAnimation(
    fig,
    animate,
    frames=72,
    interval=50,
    blit=True
)

plt.show()
