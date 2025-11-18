#version 330
in float in_posx;
in int in_letter;
out int v_letter;

void main() {
	v_letter = in_letter;
	gl_Position = vec4(in_posx, 0.0, 0.0, 1.0);
}
