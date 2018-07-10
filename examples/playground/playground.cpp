// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

// Hardware dependency
#include <librealsense2/rs.hpp>

// RealSense SDK helpers
#include "../example.hpp"
#include <imgui.h>
#include "imgui_impl_glfw.h"

// Standard
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <chrono>

float get_depth_scale(rs2::device dev)
{
    // Go over the device's sensors
    for (rs2::sensor& sensor : dev.query_sensors())
    {
        // Check if the sensor if a depth sensor
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
        {
            return dpt.get_depth_scale();
        }
    }
    throw std::runtime_error("Device does not have a depth sensor");
}

int main(int argc, char * argv[]) try
{
    printf("my playground\n");
    // parameters
    int W_win = 640;
    int H_win = 480;
    float depth_clipping_distance = 1.0f;
    // UI init
    window app(W_win, H_win, "My playground"); // Simple window handling
    ImGui_ImplGlfw_Init(app, false);           // ImGui library intializition
    texture renderer;                          // Helper for renderig images
    // Device init
    rs2::pipeline pipe;
    rs2::pipeline_profile profile = pipe.start();
    float depth_scale = get_depth_scale(profile.get_device());

    int count = 0;
    while (app) // Application still alive?
    {
        printf("[%d]\n", count);
        rs2::frameset frameset = pipe.wait_for_frames();
        rs2::video_frame color_frame = frameset.get_color_frame();
        if (!color_frame)
        {
            continue;
        }

        float w = static_cast<float>(app.width());
        float h = static_cast<float>(app.height());
        rect color_frame_rect{ 0, 0, w, h };
        color_frame_rect = color_frame_rect.adjust_ratio({ static_cast<float>(color_frame.get_width()),static_cast<float>(color_frame.get_height()) });

        renderer.render(color_frame, color_frame_rect);

        ++count;
    }
    system("pause");
    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}