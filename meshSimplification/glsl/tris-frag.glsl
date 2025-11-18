#version 330

uniform vec3 u_light_pos; // light position in view coordinates
uniform vec4 u_color; // k_d material parameter, otherwise, the color to draw if lighting disabled
uniform bool u_use_lighting;

in vec3 g_vert; // vertex position in view coordinates
in vec3 g_norm; // face normal as computed by geometry shader

out vec4 f_color;

const float LIGHT_AMBIENT = 0.3;           // Ambient light intensity
const vec3 LIGHT = vec3( 0.8, 0.8, 0.8 );  // Light intensity
const vec3 k_s   = vec3( 1, 1, 1);		   // specular material parameter
const float SHININESS = 50.0;			   // shininess exponent for specular

void main() {
	if ( u_use_lighting == false ) {
		f_color = u_color; 
		return;
	}

	// Setup vectors for computing lighting, and flip the normal if the face is backfacing
	// Note that all these vectors are in view coordinates
	vec3 normal_vector = normalize( g_norm ) * (gl_FrontFacing ? 1 : -1);
	vec3 light_vector = normalize( u_light_pos - g_vert );
	vec3 view_vector = normalize( - g_vert ); 
	vec3 half_vector = normalize( light_vector + view_vector );

	// Compute lighting contributions
	float cos_theta = dot( light_vector, normal_vector );
	vec3 Ld = u_color.xyz * LIGHT * max( cos_theta, 0.0 );	
	vec3 Ls = k_s * LIGHT *  pow( max( dot( half_vector, normal_vector ), 0.0 ), SHININESS );
	vec3 La = u_color.xyz * LIGHT_AMBIENT;

	f_color = vec4(Ld + Ls + La, u_color.a);
}