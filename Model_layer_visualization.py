from collections import defaultdict
color_map = defaultdict(dict)
#color_map[Conv2D]['fill'] = '#00f5d4'
color_map[Dropout]['fill'] = 'grey'

import visualkeras
from PIL import ImageFont

font = ImageFont.truetype("/content/AGENCYB.TTF", 21)
'''visualkeras.layered_view(model).show() # display using your system viewer
visualkeras.layered_view(model, to_file='output.png') # write to disk'''
visualkeras.layered_view(model1, legend=True, font=font, to_file='output.png', color_map=color_map).show()