#version 330

uniform mat4 u_mv;
uniform mat4 u_mvp;

in vec3 in_position;

out vec3 v_vert;

void main() {
	gl_Position = u_mvp * vec4(in_position, 1.0);
	v_vert = (u_mv * vec4(in_position, 1.0)).xyz;
}