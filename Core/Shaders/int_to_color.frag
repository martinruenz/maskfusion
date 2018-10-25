/*
 * This file is part of https://github.com/martinruenz/maskfusion
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 */

#version 330 core

#include "color_table.glsl"

in vec2 texcoord;
out vec3 FragColor;
uniform usampler2D gSampler;

void main()
{
    int id = int(texture(gSampler, texcoord.xy));
    FragColor = (id == 255) ? vec3(1,1,1) : vec3(colors[int(mod(id,numColors))]);
}
