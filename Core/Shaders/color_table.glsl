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

#if 0
// Color lookup table
const int numColors = 31;
const vec3 colors[numColors]=vec3[numColors](
    vec3(0.1,0.1,0.1),
    vec3(0,0,1),
    vec3(1,0,0),
    vec3(0,1,0),
    vec3(1,0.10,0.72),
    vec3(1,0.82,0),
    vec3(0,0.51,0.96),
    vec3(0,0.55,0.27),
    vec3(0.65,0.37,0.24),
    vec3(0.31,0,0.41),
    vec3(0,1,0.96),
    vec3(0.24,0.48,0.55),
    vec3(0.93,0.65,1),
    vec3(0.82,1,0.58),
    vec3(0.72,0.31,1),
    vec3(0.89,0.10,0.34),
    vec3(0.51,0.51,0),
    vec3(0,1,0.58),
    vec3(0.37,0,0.17),
    vec3(0.96,0.51,0.06),
    vec3(0.79,1,0),
    vec3(0.17,0.24,0),
    vec3(0,0.20,0.75),
    vec3(1,0.79,0.51),
    vec3(0,0.17,0.37),
    vec3(0.62,0.44,0.55),
    vec3(0.31,0.72,0.06),
    vec3(0.62,0.75,1),
    vec3(0.58,0.62,0.48),
    vec3(1,0.48,0.68),
    vec3(0.62,0.03,0)
);
#endif

// used for MS-COCO
const int numColors = 80;
const vec3 colors[numColors]=vec3[numColors](
        vec3(0.1,0.1,0.1),
        vec3(0,0,1),
        vec3(1,0,0),
        vec3(0,1,0),
        vec3(0,1,0),
        vec3(1,0.10,0.72),
        vec3(1,0.82,0),
        vec3(0,0.51,0.96),
        vec3(0,0.55,0.27),
        vec3(0.65,0.37,0.24),
        vec3(0.31,0,0.41),
        vec3(0,1,0.96),
        vec3(0.24,0.48,0.55),
        vec3(0.93,0.65,1),
        vec3(0.82,1,0.58),
        vec3(0.72,0.31,1),
        vec3(0.89,0.10,0.34),
        vec3(0.51,0.51,0),
        vec3(0,1,0.58),
        vec3(0.37,0,0.17),
        vec3(0.96,0.51,0.06),
        vec3(0.79,1,0),
        vec3(0.17,0.24,0),
        vec3(0,0.20,0.75),
        vec3(1,0.79,0.51),
        vec3(0,0.17,0.37),
        vec3(0.62,0.44,0.55),
        vec3(0.31,0.72,0.06),
        vec3(0.62,0.75,1),
        vec3(0.58,0.62,0.48),
        vec3(1,0.48,0.68),
        vec3(0.62,0.03,0),
        vec3(0.1,0.1,0.1),
        vec3(0.000, 0.000, 1.000), // ball
        vec3(1,0,0),
        vec3(0,1,0),
        vec3(1,0.10,0.72),
        vec3(1,0.82,0),
        vec3(0,0.51,0.96),
        vec3(0,0.55,0.27),
        vec3(0.65,0.37,0.24),
        vec3(0.31,0,0.41),
        vec3(0,1,0.96),
        vec3(0.24,0.48,0.55),
        vec3(0.93,0.65,1),
        vec3(0.82,1,0.58),
        vec3(0.72,0.31,1),
        vec3(1,1,0), // banana
        vec3(0.486, 0.988, 0.000), // apple
        vec3(0,1,0.58),
        vec3(1.000, 0.843, 0.000), // orange
        vec3(0.96,0.51,0.06),
        vec3(1.000, 0.549, 0.000), // carrot
        vec3(0.17,0.24,0),
        vec3(0.647, 0.165, 0.165), // pizza
        vec3(0.980, 0.922, 0.843), // donut
        vec3(0,0.17,0.37),
        vec3(0.62,0.44,0.55),
        vec3(0.31,0.72,0.06),
        vec3(0.62,0.75,1),
        vec3(0.58,0.62,0.48),
        vec3(1,0.48,0.68),
        vec3(0.62,0.03,0),
        vec3(0.51,0.51,0),
        vec3(0,1,0.58),
        vec3(0.37,0,0.17),
        vec3(0.79,1,0),
        vec3(1.000, 0.2, 0.1), // keyboard
        vec3(0.17,0.24,0),
        vec3(0,0.20,0.75),
        vec3(1,0.79,0.51),
        vec3(0,0.17,0.37),
        vec3(0.62,0.44,0.55),
        vec3(0.31,0.72,0.06),
        vec3(0.62,0.75,1),
        vec3(1.000, 1.000, 0.000), // clock
        vec3(1,0.48,0.68),
        vec3(0.62,0.03,0),
        vec3(0.133, 0.545, 0.133), // teddy
        vec3(0.31,0.72,0.06)
);
