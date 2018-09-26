cimport numpy as np

def set_texture(tex_writer, int point_cloud_size):
    cdef int pnt_idx
    for pnt_idx in range(point_cloud_size):
        tex_writer.setData2f(-0.5, +0.5)
        tex_writer.setData2f(-0.5, -0.5)
        tex_writer.setData2f(+0.5, -0.5)
        tex_writer.setData2f(+0.5, +0.5)
        tex_writer.setData2f(-0.5, +0.5)
        tex_writer.setData2f(+0.5, -0.5)


def load_point_cloud(pos_writer, color_writer, np.ndarray[np.float64_t, ndim=2] point_cloud,
                     np.ndarray[np.float64_t, ndim=2] color, int point_cloud_size, int prev_point_cloud_size):
    pos_writer.set_row(0)
    color_writer.set_row(0)
    color[color > 1] = 1
    cdef int total_points = point_cloud.shape[0]
    for pnt_idx in range(min(point_cloud_size, max(total_points, prev_point_cloud_size))):
        if pnt_idx < total_points:
            for i in range(6):
                pos_writer.setData3f(-float(point_cloud[pnt_idx, 0]), float(point_cloud[pnt_idx, 1]), float(point_cloud[pnt_idx, 2]))
                if color is not None:
                    color_writer.setData4(color[pnt_idx, 0], color[pnt_idx, 1], color[pnt_idx, 2], color[pnt_idx, 3])
        else:
            for i in range(6):
                color_writer.setData4(0., 0., 0., 0.)

def update_bb(cuboid, pos_writer, color_writer, tuple size, tuple translation,
              tuple rotation, tuple color, str text, text_node, text_np):
    cdef float wx, wy, wz
    wx, wy, wz = size
    # wx, wy, wz = size
    pos_writer.setRow(0)
    color_writer.setRow(0)
    pos_writer.setData3f(-0.5 * wx, -0.5 * wy, 0.)
    pos_writer.setData3f(-0.5 * wx, -0.5 * wy, wz)
    pos_writer.setData3f(0.5 * wx, -0.5 * wy, wz)
    pos_writer.setData3f(0.5 * wx, -0.5 * wy, 0.)
    pos_writer.setData3f(-0.5 * wx, 0.5 * wy, 0.)
    pos_writer.setData3f(-0.5 * wx, 0.5 * wy, wz)
    pos_writer.setData3f(0.5 * wx, 0.5 * wy, wz)
    pos_writer.setData3f(0.5 * wx, 0.5 * wy, 0.)
    cdef int i = 0
    cdef float r, g, b, a
    r, g, b, a = color
    for i in range(8):
        color_writer.setData4f(r, g, b, a)
    cdef float tx, ty, tz
    tx, ty, tz = translation
    ty = -ty
    cdef float rx, ry, rz
    rx, ry, rz = rotation
    rz = -rz
    cuboid.setPos(ty, tx, tz)
    cuboid.setHpr(-rz - 90., ry, rx)
    text_node.setText(text)
    text_np.setPos((0., 0., wz+0.2))
    text_np.setHpr(rz + 90., 0, 0)