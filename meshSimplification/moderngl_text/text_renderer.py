import numpy as np
import moderngl as mgl
from pathlib import Path
from PIL import Image
from pyglm import glm


class TextRenderer:
    def __init__(self, ctx: mgl.Context):
        current_dir = Path(__file__).parent
        self.prog = ctx.program(
            vertex_shader=open(current_dir / 'glsl' / 'font-vert.glsl').read(),
            geometry_shader=open(current_dir / 'glsl' / 'font-geom.glsl').read(),
            fragment_shader=open(current_dir / 'glsl' / 'font-frag.glsl').read())

        im = Image.open(current_dir / 'data' / 'font-atlas.png')
        atlas = np.array(im)
        atlas_texture = ctx.texture(atlas.shape[::-1], 1, atlas.tobytes())
        atlas_texture.filter = (mgl.LINEAR, mgl.LINEAR)
        atlas_texture.use(0)  # bind to texture unit 0
        self.prog['u_texture'].value = 0  # texture unit 0

        # maximum string length will be 256 characters
        posx = np.array(range(256), dtype='f4')
        posxbo = ctx.buffer(posx.tobytes())
        self.letter = np.zeros(256, dtype='i4')
        self.letterbo = ctx.buffer(self.letter.tobytes())
        vao_content = [(posxbo, 'f', 'in_posx'), (self.letterbo, 'i', 'in_letter')]  # 1 float and one int per vertex
        self.vao_text = ctx.vertex_array(self.prog, vao_content, mode=mgl.POINTS)

    def render_text(self, s: str, P: glm.mat4, V: glm.mat4, pos: glm.vec3 = glm.vec3(0, 0, 0), char_width: float = 0.05,
                    color: glm.vec4 = glm.vec4(1, 1, 1, 1), centered: bool = True, view_aligned=True):
        self.prog['u_proj'].write(P)
        self.prog['u_view'].write(V)
        self.prog['u_position'].write(pos)
        self.prog['u_width'].value = char_width
        self.prog['u_height'].value = char_width * 2.0
        self.prog['u_color'].value = color
        if len(s) > 256:
            s = s[:256]  # truncate to max length
        if centered:  # offset in character units
            self.prog['u_char_offset'].value = (-len(s) * 0.5, +0.5)
        else:
            self.prog['u_char_offset'].value = (0.0, 0.0)
        self.prog['u_view_aligned'].value = view_aligned
        self.letter[:len(s)] = [ord(c) for c in s]
        self.letterbo.write(self.letter.tobytes())
        self.vao_text.render(mode=mgl.POINTS, vertices=len(s))
