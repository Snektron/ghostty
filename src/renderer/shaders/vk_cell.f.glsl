#version 450 core

layout(location = 1) in vec2 glyph_tex_coords;
layout(location = 3) flat in uint mode;

// The color for this cell. If this is a background pass this is the
// background color. Otherwise, this is the foreground color.
layout(location = 0) flat in vec4 color;

// The position of the cells top-left corner.
layout(location = 2) flat in vec2 screen_cell_pos;

// Position the fragment coordinate to the upper left
layout(origin_upper_left) in vec4 gl_FragCoord;

// Must declare this output for some versions of OpenGL.
layout(location = 0) out vec4 out_FragColor;

// Font texture
layout(set = 0, binding = 0) uniform sampler2D text;
layout(set = 0, binding = 1) uniform sampler2D text_color;

// Note: Keep in sync with CellPipeline.zig!
layout(push_constant, std430) uniform Uniforms {
     mat4 projection;
     vec4 grid_padding;
     vec2 grid_size;
     vec2 cell_size;
     float min_contrast;
     uint padding_vertical_top;
     uint padding_vertical_bottom;
};

// See vertex shader
const uint MODE_BG = 1u;
const uint MODE_FG = 2u;
const uint MODE_FG_CONSTRAINED = 3u;
const uint MODE_FG_COLOR = 7u;
const uint MODE_FG_POWERLINE = 15u;

void main() {
    float a;

    switch (mode) {
    case MODE_BG:
        out_FragColor = color;
        break;

    case MODE_FG:
    case MODE_FG_CONSTRAINED:
    case MODE_FG_POWERLINE:
        a = texture(text, glyph_tex_coords).r;
        vec3 premult = color.rgb * color.a;
        out_FragColor = vec4(premult.rgb*a, a);
        break;

    case MODE_FG_COLOR:
        out_FragColor = texture(text_color, glyph_tex_coords);
        break;
    }
}
