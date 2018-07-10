// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

// RealSense SDK
#include <librealsense2/rs.hpp>
#include <librealsense2/hpp/rs_internal.hpp>

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

rs2::stream_profile get_color_stream_profile(const std::vector<rs2::stream_profile>& streams)
{
    for (rs2::stream_profile sp : streams)
    {
        rs2_stream profile_stream = sp.stream_type();
        if (profile_stream == RS2_STREAM_COLOR)
        {
            return sp;
        }
    }
    printf("can not find a color stream!\n");
    throw std::runtime_error("No color stream available");
}

void update_sw_sensor(rs2::software_sensor& sensor, rs2::stream_profile& profile, const int& frame_number, const int& w, const int& h, const int& bpp)
{
    uint8_t* p_data = new uint8_t[w*h*bpp];
    memset(p_data, 0, w*h*bpp);

    sensor.on_video_frame(
        { 
            (void*)p_data,   // Frame pixels from capture API
            [](void*) {},    // Custom deleter (if required)
            w*bpp , bpp,     // Stride and Bytes-per-pixel
            (rs2_time_t)frame_number * 16, RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK, frame_number, // Timestamp, Frame# for potential sync services
            profile 
        }
    );
}

int main(int argc, char * argv[]) try
{
    printf("my playground\n");
    
    // Parameters
    int W_win = 640;
    int H_win = 480;
    float depth_clipping_distance = 1.0f;
    int W_normal = W_win;
    int H_normal = H_win;
    int BPP_normal = 3;
    
    // UI init
    window app(W_win, H_win, "My playground"); // Simple window handling
    ImGui_ImplGlfw_Init(app, false);           // ImGui library intializition
    texture renderer;                          // Helper for renderig images
    
    // Device init
    rs2::pipeline pipe;
    rs2::pipeline_profile profile = pipe.start();
    rs2::stream_profile color_stream_profile = get_color_stream_profile(profile.get_streams());
    rs2::video_stream_profile color_video_stream_profile = color_stream_profile.as<rs2::video_stream_profile>();
    float depth_scale = get_depth_scale(profile.get_device());
    
    rs2::software_device dev; // Create software-only device
    auto normal_map_sensor = dev.add_sensor("NormalMap"); // Define single sensor
    // Use color stream profile for proper RGB channel settings, however this stream is use to display normal map
    auto normal_map_stream = normal_map_sensor.add_video_stream({
        RS2_STREAM_COLOR, 0, 1,
        W_normal, H_normal, 60,
        BPP_normal,
        RS2_FORMAT_RGB8, color_video_stream_profile.get_intrinsics() }
    );
    rs2::syncer sync;
    normal_map_sensor.open(normal_map_stream);
    normal_map_sensor.start(sync);

    int frame_number = 0;
    while (app) // Application still alive?
    {
        printf("[%d]\n", frame_number);
        rs2::frameset frameset = pipe.wait_for_frames();
        rs2::video_frame color_frame = frameset.get_color_frame();
        if ( !color_frame)
        {
            continue;
        }

        // manually submit sw frame to sw sensor
        update_sw_sensor(normal_map_sensor, normal_map_stream, frame_number, W_normal, H_normal, BPP_normal);

        rs2::frameset frameset_sw = sync.wait_for_frames();
        rs2::video_frame normal_frame = frameset_sw.get_color_frame();
        if ( !normal_frame )
        {
            continue;
        }

        float w = static_cast<float>(app.width());
        float h = static_cast<float>(app.height());
        rect frame_rect{ 0, 0, w, h };
        frame_rect = frame_rect.adjust_ratio({ static_cast<float>(normal_frame.get_width()),static_cast<float>(normal_frame.get_height()) });

        renderer.render(normal_frame, frame_rect);

        ++frame_number;
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