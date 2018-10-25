/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 *
 * The use of the code within this file and all code within files that
 * make up the software that is ElasticFusion is permitted for
 * non-commercial purposes only.  The full terms and conditions that
 * apply to the code within this file are detailed within the LICENSE.txt
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/>
 * unless explicitly stated.  By downloading this file you agree to
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

// #version 330 core
#version 430 core

layout (location = 0) in vec4 position;
layout (location = 1) in vec4 color;
layout (location = 2) in vec4 normal;

layout(std430, binding=0) buffer bb_block { int bounding_box[6]; };

uniform mat4 MVP;
uniform float threshold;
uniform int colorType;
uniform int unstable;
uniform int drawWindow;
uniform int time;
uniform int timeDelta;
uniform uint maskID;
uniform uint classID;
uniform int highlighted;
//uniform mat4 pose;

out vec4 vColor;
out vec4 vPosition;
out vec4 vNormRad;
out mat4 vMVP;
out int vTime;
out int colorType0;
out int drawWindow0;
out int timeDelta0;
out uint maskID0;
out uint classID0;
out int vHighlighted;

void main()
{
    if(position.w > threshold || unstable == 1)
    {
        colorType0 = colorType;
        drawWindow0 = drawWindow;
        vColor = color;
        vPosition = position;
        vNormRad = normal;
        vMVP = MVP;
        vTime = time;
        vHighlighted = highlighted;
        timeDelta0 = timeDelta;
        maskID0 = maskID;
        classID0 = classID;
        gl_Position = MVP * vec4(position.xyz, 1.0);

        const float bbscale = 1000;
        int x_mm = int(bbscale * position.x);
        int y_mm = int(bbscale * position.y);
        int z_mm = int(bbscale * position.z);
        atomicMin(bounding_box[0], x_mm);
        atomicMin(bounding_box[1], y_mm);
        atomicMin(bounding_box[2], z_mm);
        atomicMax(bounding_box[3], x_mm);
        atomicMax(bounding_box[4], y_mm);
        atomicMax(bounding_box[5], z_mm);
    }
    else
    {
        colorType0 = -1;
    }
}
