
from os import path
import numpy as np
from vispy import app, gloo
from vispy.util.transforms import perspective, translate, rotate
from vispy.io import load_data_file, read_mesh, load_crate

VERT_COLOR_CODE = """
// Uniforms
// ------------------------------------
uniform   mat4 u_model;
uniform   mat4 u_view;
uniform   mat4 u_projection;
uniform   vec4 u_color;

// Attributes
// ------------------------------------
attribute vec3 a_position;
attribute vec4 a_color;
attribute vec3 a_normal;

// Varying
// ------------------------------------
varying vec4 v_color;

void main()
{
    v_color = u_color;
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
}
"""


FRAG_COLOR_CODE = """
// Varying
// ------------------------------------
varying vec4 v_color;

void main()
{
    gl_FragColor = v_color;
}
"""

VERT_TEX_CODE = """
uniform   mat4 u_model;
uniform   mat4 u_view;
uniform   mat4 u_projection;

attribute vec3 a_position;
attribute vec2 a_texcoord;

varying vec2 v_texcoord;

void main()
{
  v_texcoord = a_texcoord;
  gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
}
"""


FRAG_TEX_CODE = """
uniform sampler2D u_texture;
varying vec2 v_texcoord;

void main()
{
  float ty = v_texcoord.y;
  float tx = sin(ty*50.0)*0.01 + v_texcoord.x;
  gl_FragColor = texture2D(u_texture, vec2(tx, ty));
}
"""



class Canvas(app.Canvas):

  def __init__(self):
    app.Canvas.__init__(self, keys='interactive', size=(800, 600))

    dirname = path.join(path.abspath(path.curdir),'data')
    positions, faces, normals, texcoords = \
      read_mesh(load_data_file('cube.obj', directory=dirname))

    self.filled_buf = gloo.IndexBuffer(faces)

    if False:
      self.program = gloo.Program(VERT_TEX_CODE, FRAG_TEX_CODE)
      self.program['a_position'] = gloo.VertexBuffer(positions)
      self.program['a_texcoord'] = gloo.VertexBuffer(texcoords)
      self.program['u_texture'] = gloo.Texture2D(load_crate())
    else:
      self.program = gloo.Program(VERT_COLOR_CODE, FRAG_COLOR_CODE)
      self.program['a_position'] = gloo.VertexBuffer(positions)
      self.program['u_color'] = 1, 0, 0, 1

    self.view = translate((0, 0, -5))
    self.model = np.eye(4, dtype=np.float32)

    gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
    self.projection = perspective(45.0, self.size[0] /
                                  float(self.size[1]), 2.0, 10.0)

    self.program['u_projection'] = self.projection

    self.program['u_model'] = self.model
    self.program['u_view'] = self.view

    self.theta = 0
    self.phi = 0

    gloo.set_clear_color('gray')
    gloo.set_state('opaque')
    gloo.set_polygon_offset(1, 1)

    self._timer = app.Timer('auto', connect=self.on_timer, start=True)

    self.show()

  # ---------------------------------
  def on_timer(self, event):
    self.theta += .5
    self.phi += .5
    self.model = np.dot(rotate(self.theta, (0, 1, 0)),
                        rotate(self.phi, (0, 0, 1)))
    self.program['u_model'] = self.model
    self.update()

  # ---------------------------------
  def on_resize(self, event):
    gloo.set_viewport(0, 0, event.physical_size[0], event.physical_size[1])
    self.projection = perspective(45.0, event.size[0] /
                                  float(event.size[1]), 2.0, 10.0)
    self.program['u_projection'] = self.projection

  # ---------------------------------
  def on_draw(self, event):
    gloo.clear()

    # Filled cube

    gloo.set_state(blend=False, depth_test=True, polygon_offset_fill=True)
    self.program.draw('triangles', self.filled_buf)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
  c = Canvas()
  app.run()
