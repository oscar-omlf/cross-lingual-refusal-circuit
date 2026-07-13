import numpy as np
from matplotlib import colormaps



EDGE_TYPE_COLORS = {
    'q': "#FF00FF", # Purple
    'k': "#00FF00", # Green
    'v': "#0000FF", # Blue
    None: "#000000", # Black
}

def generate_random_color(colorscheme: str) -> str:
    """
    https://stackoverflow.com/questions/28999287/generate-random-colors-rgb
    """

    def rgb2hex(rgb):
        """
        https://stackoverflow.com/questions/3380726/converting-an-rgb-color-tuple-to-a-hexidecimal-string
        """
        return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

    rgba = colormaps[colorscheme](np.random.random())
    rgb = tuple(round(channel * 255) for channel in rgba[:3])
    return rgb2hex(rgb)
    
