#version 330

layout (lines) in;
layout (triangle_strip, max_vertices = 24) out;

in vec3 v_position[];

out vec3 g_endpoint_a;
out vec3 g_endpoint_b;
out vec3 g_view_pos;

uniform float u_radius;
uniform mat4 u_view;
uniform mat4 u_proj;

void main() {
    // Get the two endpoints of the line
    vec3 a = v_position[0];
    vec3 b = v_position[1];
    float r = u_radius; // could use values specified at verts alternatively

    // Transform endpoints to view space
    vec3 a_view = (u_view * vec4(a, 1.0)).xyz;
    vec3 b_view = (u_view * vec4(b, 1.0)).xyz;
    
     // Compute oriented bounding box in view space
    vec3 axis = normalize(b_view - a_view);
    vec3 center = (a_view + b_view) * 0.5;
    float half_length = length(b_view - a_view) * 0.5 + r;  // Extend by radius for caps
    
    // Create orthogonal basis for the capsule
    // Use view direction (Z) to get proper billboard orientation
    vec3 view_dir = vec3(0.0, 0.0, 1.0);
    vec3 right = cross(view_dir, axis);
    
    // Handle case where axis is parallel to view direction
    if (length(right) < 0.001) {
        right = vec3(1.0, 0.0, 0.0);
    } else {
        right = normalize(right);
    }
    
    vec3 up = normalize(cross(axis, right));
    
    // Scale basis vectors by radius
    right *= r;
    up *= r;
    vec3 forward = axis * half_length;
    
// Compute 8 corners of the oriented box
    vec3 corners[8];
    corners[0] = center - forward - right - up;  // back-left-bottom
    corners[1] = center - forward + right - up;  // back-right-bottom
    corners[2] = center - forward - right + up;  // back-left-top
    corners[3] = center - forward + right + up;  // back-right-top
    corners[4] = center + forward - right - up;  // front-left-bottom
    corners[5] = center + forward + right - up;  // front-right-bottom
    corners[6] = center + forward - right + up;  // front-left-top
    corners[7] = center + forward + right + up;  // front-right-top
    
    // Pass through to fragment shader
    g_endpoint_a = a_view;
    g_endpoint_b = b_view;
    
    // Emit 6 faces as triangle strips (4 vertices each = 2 triangles)
    // Winding order ensures outward-facing normals for back-face culling
    
    // Front face (+forward direction) - looking at it from front
    int front_face[4] = int[4](4, 6, 5, 7);
    for (int i = 0; i < 4; i++) {
        gl_Position = u_proj * vec4(corners[front_face[i]], 1.0);
        g_view_pos = corners[front_face[i]];
        EmitVertex();
    }
    EndPrimitive();
    
    // Back face (-forward direction) - looking at it from back
    int back_face[4] = int[4](0, 1, 2, 3);
    for (int i = 0; i < 4; i++) {
        gl_Position = u_proj * vec4(corners[back_face[i]], 1.0);
        g_view_pos = corners[back_face[i]];
        EmitVertex();
    }
    EndPrimitive();
    
    // Right face (+right direction) - looking at it from right
    int right_face[4] = int[4](1, 5, 3, 7);
    for (int i = 0; i < 4; i++) {
        gl_Position = u_proj * vec4(corners[right_face[i]], 1.0);
        g_view_pos = corners[right_face[i]];
        EmitVertex();
    }
    EndPrimitive();
    
    // Left face (-right direction) - looking at it from left
    int left_face[4] = int[4](4, 0, 6, 2);
    for (int i = 0; i < 4; i++) {
        gl_Position = u_proj * vec4(corners[left_face[i]], 1.0);
        g_view_pos = corners[left_face[i]];
        EmitVertex();
    }
    EndPrimitive();
    
    // Top face (+up direction) - looking at it from top
    int top_face[4] = int[4](2, 3, 6, 7);
    for (int i = 0; i < 4; i++) {
        gl_Position = u_proj * vec4(corners[top_face[i]], 1.0);
        g_view_pos = corners[top_face[i]];
        EmitVertex();
    }
    EndPrimitive();
    
    // Bottom face (-up direction) - looking at it from bottom
    int bottom_face[4] = int[4](4, 5, 0, 1);
    for (int i = 0; i < 4; i++) {
        gl_Position = u_proj * vec4(corners[bottom_face[i]], 1.0);
        g_view_pos = corners[bottom_face[i]];
        EmitVertex();
    }
    EndPrimitive();
}