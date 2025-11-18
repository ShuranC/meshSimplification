 #version 330
uniform sampler2D u_texture;
uniform vec4 u_color = vec4(1,1,1,1);
uniform float smoothness = 1.0;
in vec2 g_texcoord;
out vec4 f_color;

void main() {
	float alpha = texture(u_texture, g_texcoord).r;
	vec2 dtex = dFdx(g_texcoord);
	vec2 dtexdy = dFdy(g_texcoord);
	float max_dtex = max(length(dtex), length(dtexdy));
	// compute smoothness based on screen space derivative
	// arbitrary extra factor here let smoothness be a reasonable valueting, that can be adjusted
	float s = min(127.0, max_dtex * 10000.0 * smoothness); 
	float a = smoothstep(127.0 + s, 127.0 - s, alpha*255.0);
	if ( a < 0.001 ) discard;
	f_color = vec4(1.0, 1.0, 1.0, a) * u_color;
}