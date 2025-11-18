#version 330

in vec3 g_endpoint_a;
in vec3 g_endpoint_b;
in vec3 g_view_pos;

out vec4 f_color;

uniform float u_radius;
uniform mat4 u_proj;
uniform vec3 u_light_pos;
uniform mat4 u_view;
uniform vec4 u_color;

const float LIGHT_AMBIENT = 0.3;           // Ambient light intensity
const vec3 LIGHT = vec3( 0.8, 0.8, 0.8 );  // Light intensity
const vec3 k_s   = vec3( 1, 1, 1);		   // specular material parameter
const float SHININESS = 50.0;			   // shininess exponent for specular

bool intersect_cylinder(vec3 o, vec3 d, vec3 a, vec3 b, float r, out float t, out vec3 normal, out float s) {
    // Cylinder between a and b with radius r
    vec3 axis = b - a;
    float axis_length = length(axis);
    vec3 axis_norm = axis / axis_length;
    
    vec3 oa = o - a;
    
    // Project ray onto plane perpendicular to axis
    vec3 d_perp = d - dot(d, axis_norm) * axis_norm;
    vec3 oa_perp = oa - dot(oa, axis_norm) * axis_norm;
    
    // Solve quadratic: |oa_perp + t * d_perp|^2 = r^2
    float a_coef = dot(d_perp, d_perp);
    
    // Check if ray is (nearly) parallel to axis
    if (a_coef < 0.0001) {
        // Ray parallel to cylinder - check if we're inside the radius
        float dist_to_axis = length(oa_perp);
        if (dist_to_axis <= r) {
            // We're inside the cylinder, find entry point along axis
            t = 0.001;  // Small positive value
            vec3 p = o + t * d;
            s = dot(p - a, axis_norm) / axis_length;
            if (s >= 0.0 && s <= 1.0) {
                vec3 closest = a + s * axis;
                normal = normalize(p - closest);
                return true;
            }
        }
        return false;
    }
    
    float b_coef = 2.0 * dot(oa_perp, d_perp);
    float c_coef = dot(oa_perp, oa_perp) - r * r;
    
    float discriminant = b_coef * b_coef - 4.0 * a_coef * c_coef;
    
    if (discriminant < 0.0) {
        return false;
    }
    
    float sqrt_disc = sqrt(discriminant);
    float t1 = (-b_coef - sqrt_disc) / (2.0 * a_coef);
    float t2 = (-b_coef + sqrt_disc) / (2.0 * a_coef);
    
    t = (t1 > 0.0) ? t1 : t2;
    
    if (t <= 0.0) {
        return false;
    }
    
    // Check if intersection is within cylinder segment
    vec3 p = o + t * d;
    s = dot(p - a, axis_norm) / axis_length;
    
    if (s >= 0.0 && s <= 1.0) {
        // Valid cylinder intersection
        vec3 closest = a + s * axis;
        normal = normalize(p - closest);
        return true;
    }
    
    return false;
}

bool intersect_sphere(vec3 o, vec3 d, vec3 center, float r, out float t, out vec3 normal) {
    vec3 oc = o - center;
    float a_coef = dot(d, d);
    float b_coef = 2.0 * dot(oc, d);
    float c_coef = dot(oc, oc) - r * r;
    
    float discriminant = b_coef * b_coef - 4.0 * a_coef * c_coef;
    
    if (discriminant < 0.0) {
        return false;
    }
    
    float sqrt_disc = sqrt(discriminant);
    float t1 = (-b_coef - sqrt_disc) / (2.0 * a_coef);
    float t2 = (-b_coef + sqrt_disc) / (2.0 * a_coef);
    
    t = (t1 > 0.0) ? t1 : t2;
    
    if (t <= 0.0) {
        return false;
    }
    
    vec3 p = o + t * d;
    normal = normalize(p - center);
    return true;
}

void main() {
    // Ray in view space: origin at (0,0,0), direction toward fragment
    vec3 ray_origin = vec3(0.0, 0.0, 0.0);
    vec3 ray_dir = normalize(g_view_pos);
    
   float t;
    vec3 normal_view;
    bool hit = false;
    float t_best = 1e10;  // Track closest hit
        
    // Try cylinder body
    float s;
    float t_cyl;
    vec3 normal_cyl;
    if (intersect_cylinder(ray_origin, ray_dir, g_endpoint_a, g_endpoint_b, u_radius, t_cyl, normal_cyl, s)) {
        if (t_cyl < t_best) {
            t_best = t_cyl;
            t = t_cyl;
            normal_view = normal_cyl;
            hit = true;
        }
    }
    
    // Check cap at endpoint a
    float t_cap_a;
    vec3 normal_cap_a;
    if (intersect_sphere(ray_origin, ray_dir, g_endpoint_a, u_radius, t_cap_a, normal_cap_a)) {
        if (t_cap_a < t_best) {
            t_best = t_cap_a;
            t = t_cap_a;
            normal_view = normal_cap_a;
            hit = true;
        }
    }
    
    // Check cap at endpoint b
    float t_cap_b;
    vec3 normal_cap_b;
    if (intersect_sphere(ray_origin, ray_dir, g_endpoint_b, u_radius, t_cap_b, normal_cap_b)) {
        if (t_cap_b < t_best) {
            t_best = t_cap_b;
            t = t_cap_b;
            normal_view = normal_cap_b;
            hit = true;
        }
    }
    
    if (!hit) {
        // f_color = vec4(0.5, 0.5, 0.5, 0.5); // instead show extent to debug
        // return;
        discard;
    }
    
    // Compute view space intersection point
    vec3 view_pos = ray_origin + t * ray_dir;
    
    // Compute proper depth
    vec4 clip_pos = u_proj * vec4(view_pos, 1.0);
    float ndc_depth = clip_pos.z / clip_pos.w;
    gl_FragDepth = (ndc_depth + 1.0) * 0.5;

    // Point light calculation
    vec3 light_vector = normalize(u_light_pos - view_pos);
    vec3 view_vector = normalize( - view_pos ); 
	vec3 half_vector = normalize( light_vector + view_vector );

    // Compute lighting contributions
	float cos_theta = dot( light_vector, normal_view );
	vec3 Ld = u_color.xyz * LIGHT * max( cos_theta, 0.0 );	
	vec3 Ls = k_s * LIGHT *  pow( max( dot( half_vector, normal_view ), 0.0 ), SHININESS );
	vec3 La = u_color.xyz * LIGHT_AMBIENT;

	f_color = vec4(Ld + Ls + La, u_color.a);
}