#version 150

// Uniform inputs
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ProjectionMatrix;
uniform vec2 view_size;

in vec4 p3d_Vertex;
in vec4 color;
in vec2 p3d_MultiTexCoord0;

out vec4 col;
out vec2 texcoord;
out float pointSize;
out vec4 position;


void main() {
    vec4 center = p3d_ModelViewMatrix * p3d_Vertex;
    pointSize = 0.05;  //center.w * 0.05 * p3d_Vertex.y / 10;
    center.xy += p3d_MultiTexCoord0 * pointSize;
    position = center;
    center = p3d_ProjectionMatrix * center;
    vec2 minSize = 1.6 / view_size * sign(p3d_MultiTexCoord0);
    center.x += minSize.x * center.w;
    center.y += minSize.y * center.w;

    gl_Position = center;
    col = color;
    texcoord = p3d_MultiTexCoord0;
}
