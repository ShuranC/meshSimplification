#version 330
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in int v_letter[]; // from vertex shader

out vec2 g_texcoord; // texture coordinates to fragment shader

uniform mat4 u_proj;
uniform mat4 u_view;

uniform vec3 u_position = vec3(0,0,0);
uniform float u_width = 0.1; // width of each character quad
uniform float u_height = 0.2; // height of each character quad
uniform vec2 u_char_offset = vec2(0,0);
uniform bool u_view_aligned = false;

const int start_code = 32; // first character in the texture atlas
const int grid_width = 19; // grid height is 5
const float char_width = 64.0; 
const float char_height = 128.0;
const float texture_width = 1216.0;
const float texture_height = 640.0;

vec2 get_char_position(int char_code) { // will return upper-left corner of the character in texture coordinates
    int index = char_code - start_code;
    int col = index % grid_width;
    int row = index / grid_width;
    float x = col * char_width / texture_width; // divide by texture width
    float y = row * char_height / texture_height;   // divide by texture height
    return vec2(x, y);
}

void main() {
	vec4 v = u_view * vec4(u_position,1.0); 
	vec4 off = (gl_in[0].gl_Position + vec4(u_char_offset,0,0)) * vec4( u_width, u_height, 0.0, 0.0 );
	vec4 dx = vec4(u_width, 0.0, 0.0, 0.0);
	vec4 dy = vec4(0.0, -u_height, 0.0, 0.0);	
	if ( !u_view_aligned ) {
		off = u_view * off;
		dx = u_view * dx;
		dy = u_view * dy;
	}
	vec2 st = get_char_position(v_letter[0])+vec2(0.0001,0.01); // add a small offset to avoid sampling artifacts
	vec2 ds = vec2(char_width/texture_width*.999, 0.0); 
	vec2 dt = vec2(0.0, char_height/texture_height*.999);
	gl_Position = u_proj * (v+off);       g_texcoord = st;       EmitVertex();
	gl_Position = u_proj * (v+off+dy);    g_texcoord = st+dt;    EmitVertex();
	gl_Position = u_proj * (v+off+dx);    g_texcoord = st+ds;    EmitVertex();
	gl_Position = u_proj * (v+off+dx+dy); g_texcoord = st+ds+dt; EmitVertex();
	EndPrimitive(); // end the triangle strip		
}