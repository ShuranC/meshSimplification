#version 330
// compute vertex normals per triangle and emit three vertices per triangle

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec3 v_vert[];  // View-space positions from vertex shader
out vec3 g_norm;    // Face normal in view space to fragment shader
out vec3 g_vert;

void main() {
    vec3 p0 = v_vert[0];
    vec3 p1 = v_vert[1];
    vec3 p2 = v_vert[2];

    vec3 edge1 = p1 - p0;
    vec3 edge2 = p2 - p0;
    vec3 normal = normalize(cross(edge1, edge2));

    for (int i = 0; i < 3; ++i) {
        g_norm = normal;
        g_vert = v_vert[i];
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}
