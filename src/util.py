import numpy as np
import pymunk as pm


# Takes in a width and height (in pixels) of a rectangle and then returns
# a list of 4 vertices which describe the four corners
def gen_rect_verts(width: float, height: float) -> list[tuple[float, float]]:
    return [
        (- width / 2, -height / 2),
        (- width / 2, height / 2),
        (width / 2, height / 2),
        (width / 2, -height / 2),
    ]

def sigmoid(x):
    x = np.clip(x, -500, 500)  # This prevents overflow from np.exp()
    return 1 / (1 + np.exp(-x))


def vec2d_to_arr(vec: pm.Vec2d) -> list[float]:
    return [vec.x, vec.y]
