def convert_coordinates(x, y):
    scale_x = 18.25
    scale_y = -18.0104
    x_offset = 317
    y_offset = 132
    offset_y = -64
    vizdoom_x = (x - x_offset) * scale_x
    vizdoom_y = - (y - y_offset) * scale_y + offset_y
    return vizdoom_x, vizdoom_y