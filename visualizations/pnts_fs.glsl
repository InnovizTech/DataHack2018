#version 120

uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ProjectionMatrix;

in vec4 col;
in vec2 texcoord;
in float pointSize;
in vec4 position;

void main() {
    vec3 normal = vec3(0.,0.,0.);
    normal.xy = texcoord * 2.0;
    float mag = dot(normal, normal);
    if (mag > 1)
        discard;
    normal.z = sqrt(1-mag);
    vec4 lightDir = -normalize(vec4(0, 1, 0, 0)); // Light source: Above and behind left sholder
    lightDir =  p3d_ModelViewMatrix * lightDir;
    float lighting = (clamp(dot(lightDir.xyz, normal), 0., 1.) * 0.6 + 0.4);
    gl_FragColor = col;
    gl_FragColor.rgb = clamp(col.rgb, 0., 1.) * lighting;
    vec4 spherePosEye = vec4(position.xyz + normal * pointSize, 1.0);

    // convert to ClipSpace
    vec4 clipSpacePos =  p3d_ProjectionMatrix * spherePosEye;
    float depth = 0.5 * (clipSpacePos.z / clipSpacePos.w)  + 0.5;
    gl_FragDepth = depth;
}