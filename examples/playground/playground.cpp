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

class Vector3
{
	static constexpr double uZero = 1e-6;
    float mWidthRange = 1.0;
    float mHeightRange = 1.0;
public:
	float x, y, z;

	Vector3() :x(0), y(0), z(0) {};
	Vector3(float x1, float y1, float z1) :x(x1), y(y1), z(z1) {};
    Vector3(float x1, float y1, float z1, float width_range, float height_range)
        :x(x1/width_range), y(y1/height_range), z(z1),  mWidthRange(width_range), mHeightRange(height_range){};
	~Vector3() {};
	void operator=(const Vector3 &v)
	{
		x = v.x;
		y = v.y;
		z = v.z;
	};
	Vector3 operator+(const Vector3 &v)
	{
		return Vector3(x + v.x, y + v.y, z + v.z);
	};
	Vector3 operator-(const Vector3 &v) 
	{
		return Vector3(x - v.x, y - v.y, z - v.z);
	};
	float dot(const Vector3 &v) { return x * v.x + y * v.y + z * v.z; };
	float length() { return sqrtf(dot(*this)); };
    void amplifiy(const float& gain)
	{
		// x = x * gain;
		// y = y * gain;
		z = z * gain;
	};
	void normalize()
	{
		float len = length();
		if (len < uZero)
		{
			len = 1.0f;
		}
		len = 1.0f / len;

		x *= len;
		y *= len;
		z *= len;
	};
    // remap to 0.0~1.0
	void remap()
	{
		x = x * 0.5 + 0.5;
		y = y * 0.5 + 0.5;
		z = z * 0.5 + 0.5;
	};
	Vector3 crossProduct(const Vector3 &v)
	{
		return Vector3(
			y * v.z - z * v.y,
			z * v.x - x * v.z,
			x * v.y - y * v.x);
	};
	void printVec3()
	{
		std::cout << "(" << x << ", " << y << ", " << z << ")" << std::endl;
	};
};

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

rs2::stream_profile get_depth_stream_profile(const std::vector<rs2::stream_profile>& streams)
{
    for (rs2::stream_profile sp : streams)
    {
        rs2_stream profile_stream = sp.stream_type();
        if (profile_stream == RS2_STREAM_DEPTH)
        {
            return sp;
        }
    }
    printf("can not find a depth stream!\n");
    throw std::runtime_error("No depth stream available");
}

void make_normal_map(uint8_t* p_normal_data, const int& width, const int& height, const int& bpp, const rs2::depth_frame& depth_frame, float depth_scale, float clipping_dist)
{
    if( width != depth_frame.get_width() ||
        height != depth_frame.get_height()
    )
    {
        printf("size not match!\n");
        return;
    }

    const uint16_t* p_depth_frame = reinterpret_cast<const uint16_t*>(depth_frame.get_data());

    // clear normal frame
    std::memset(p_normal_data, 0, width*height*bpp);

    auto get_pixel_depth = [&](const int& _x, const int& _y, const float& gain = 1.0f)
	{
		return depth_scale * p_depth_frame[(_y * width) + _x] * gain;
	};

    static int range = 255;

    // draw normal map produced by original depthmap
     #pragma omp parallel for schedule(dynamic) //Using OpenMP to try to parallelise the loop
    for (int y = 1; y < height; y++)
    {
        for (int x = 1; x < width; x++)
        {
            // Get the depth value of the current pixel
            auto pixels_distance = get_pixel_depth(x,y);
            // Calculate the offset in other frame's buffer to current pixel
            auto offset = (y * width + x) * bpp;

            // Check if the depth value is invalid (<=0) or greater than the threashold
            if (pixels_distance <= 0.0f || pixels_distance > clipping_dist)
            {
                // Set pixel to "background" color (0x00)
                std::memset(&p_normal_data[offset], 0x00, bpp);
            }
            else
            {
                // Amplify the depth value, ohterwise the depth variation of neighbor pixels are too small,
                // and the normal vector will be (0,0,1) for most cases.
                static float depth_gain = 1000.0f;
                Vector3 t(x,     y - 1, get_pixel_depth(x,     y - 1, depth_gain));
                Vector3 l(x - 1, y,     get_pixel_depth(x - 1, y,     depth_gain));
                Vector3 c(x,     y,     get_pixel_depth(x,     y,     depth_gain));
                Vector3 normal = (l - c).crossProduct(t - c);
                
                Vector3 rgb(0,0,0);
                normal.normalize();
                normal.remap();
                rgb = Vector3(normal.x*range, normal.y*range, normal.z*range);

                // R, G, B channel:
                std::memset(&p_normal_data[offset], rgb.x, 1);
                std::memset(&p_normal_data[offset + 1], rgb.y, 1);
                std::memset(&p_normal_data[offset + 2], rgb.z, 1);
            }
        }
    }
}

void update_sw_sensor(rs2::software_sensor& sensor, rs2::stream_profile& profile, const int& frame_number, const uint8_t* p_data, const int& w, const int& h, const int& bpp)
{    
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
    int W_win = 1280;
    int H_win = 720;
    float depth_clipping_distance = 1.0f;
    int W_normal = 0;
    int H_normal = 0;
    int BPP_normal = 3;
    int W_aligned_normal = 0;
    int H_aligned_normal = 0;
    
    // UI init
    window app(W_win, H_win, "My playground"); // Simple window handling
    ImGui_ImplGlfw_Init(app, false);           // ImGui library intializition
    texture renderer;                          // Helper for renderig images
    
    // Device init
    rs2::pipeline pipe;
    rs2::pipeline_profile profile = pipe.start();

    rs2::stream_profile color_stream_profile = get_color_stream_profile(profile.get_streams());
    rs2::video_stream_profile color_video_stream_profile = color_stream_profile.as<rs2::video_stream_profile>();
    
    rs2::stream_profile depth_stream_profile = get_depth_stream_profile(profile.get_streams());
    rs2::video_stream_profile depth_video_stream_profile = depth_stream_profile.as<rs2::video_stream_profile>();
    float depth_scale = get_depth_scale(profile.get_device());
  
     // Create software-only device
    rs2::software_device dev;
    rs2::software_device dev_aligned;
    auto normal_map_sensor          = dev.add_sensor("NormalMap"); // Define single sensor
    auto normal_map_sensor_aligned  = dev_aligned.add_sensor("AlignedNormalMap");

    // Use color stream profile for proper RGB channel settings, however this stream is use to display normal map
    W_normal = depth_video_stream_profile.width();
    H_normal = depth_video_stream_profile.height();
    uint8_t* p_normal_data = new uint8_t[W_normal*H_normal*BPP_normal];
    auto normal_map_stream = normal_map_sensor.add_video_stream({
        RS2_STREAM_COLOR, 0, 0,
        W_normal, H_normal, 60,
        BPP_normal,
        RS2_FORMAT_RGB8, color_video_stream_profile.get_intrinsics() }
    );
    // aligned normal map stream
    W_aligned_normal = color_video_stream_profile.width();
    H_aligned_normal = color_video_stream_profile.height();
    uint8_t* p_aligned_normal_data = new uint8_t[W_aligned_normal*H_aligned_normal*BPP_normal];
    auto normal_map_stream_aligned = normal_map_sensor_aligned.add_video_stream({
        RS2_STREAM_COLOR, 0, 1,
        W_aligned_normal, H_aligned_normal, 60,
        BPP_normal,
        RS2_FORMAT_RGB8, color_video_stream_profile.get_intrinsics() }
    );

    // dev.create_matcher(RS2_MATCHER_DEFAULT);
    rs2::syncer sync;
    normal_map_sensor.open(normal_map_stream);
    normal_map_sensor.start(sync);
    
    rs2::syncer sync_aligned;
    normal_map_sensor_aligned.open(normal_map_stream_aligned);    
    normal_map_sensor_aligned.start(sync_aligned);    

    // normal_map_stream_aligned.register_extrinsics_to(color_video_stream_profile, { { 1,0,0,0,1,0,0,0,1 },{ 0,0,0 } });

    printf("color[%dx%d], depth[%dx%d], normal[%dx%d]\n",
        color_video_stream_profile.width(), color_video_stream_profile.height(),
        depth_video_stream_profile.width(), depth_video_stream_profile.height(),
        W_normal, H_normal
    );

    // Ohter init
    rs2::colorizer colorer;
    rs2::align aligner(RS2_STREAM_COLOR); // Depth align to Color

    int frame_number = 0;
    while (app) // Application still alive?
    {
        // printf("[%d]\n", frame_number);
        // raw data
        rs2::frameset frameset = pipe.wait_for_frames();
        rs2::video_frame color_frame = frameset.get_color_frame();
        rs2::video_frame raw_depth_frame = frameset.get_depth_frame();
        if ( !color_frame || !raw_depth_frame )
        {
            continue;
        }
        
        // normal map from aligned data
        auto processed = aligner.process(frameset);
        rs2::depth_frame aligned_depth_frame = processed.get_depth_frame();
        make_normal_map(
            p_aligned_normal_data, W_aligned_normal, H_aligned_normal, BPP_normal,
            aligned_depth_frame, depth_scale, depth_clipping_distance
        );
        update_sw_sensor(normal_map_sensor_aligned, normal_map_stream_aligned, frame_number, p_aligned_normal_data, W_aligned_normal, H_aligned_normal, BPP_normal);

        // normal map from raw data
        make_normal_map(
            p_normal_data, W_normal, H_normal, BPP_normal,
            raw_depth_frame, depth_scale, depth_clipping_distance
        );
        update_sw_sensor(normal_map_sensor, normal_map_stream, frame_number, p_normal_data, W_normal, H_normal, BPP_normal);
        
        rs2::frameset frameset_sw               = sync.wait_for_frames();
        rs2::video_frame normal_frame           = frameset_sw.get_color_frame();
        // printf("frameset_sw size(%d)\n", frameset_sw.size());
        
        rs2::frameset frameset_sw_aligned       = sync_aligned.wait_for_frames();
        rs2::video_frame aligned_normal_frame   = frameset_sw_aligned.get_color_frame();
        // printf("frameset_sw_aligned size(%d)\n", frameset_sw_aligned.size());
        if ( !normal_frame || !aligned_normal_frame )
        {
            continue;
        }

        float w = static_cast<float>(app.width());
        float h = static_cast<float>(app.height());
        
        int   showCount  = 5;
        float showOffset = 0;
        auto addShow = [&](rs2::video_frame* _vFrame, const char* _str, bool _isDepth = false)
        {
            rect frame_rect{showOffset, 0, w/showCount, h};
            // printf("showOffset[%.2f] showframe_rect[%.2f,%.2f,%.2fx%.2f]\n", showOffset, frame_rect.x, frame_rect.y, frame_rect.w, frame_rect.h);            
            frame_rect = frame_rect.adjust_ratio({ static_cast<float>(_vFrame->get_width()),static_cast<float>(_vFrame->get_height()) });
            if(_isDepth)
            {
                renderer.upload(colorer(*_vFrame));
            }
            else
            {
                renderer.upload(*_vFrame);
            }
            renderer.show(frame_rect, _str);
            showOffset += frame_rect.w;
        };

        addShow(&normal_frame,      "Normal Map");
        addShow(&raw_depth_frame,   "Depth Map", true);
        addShow(&color_frame,       "RGB");
        addShow(static_cast<rs2::video_frame*>(&aligned_normal_frame),  "Aligned Normal Map");
        addShow(static_cast<rs2::video_frame*>(&aligned_depth_frame),   "Aligned Depth Map", true);

        ++frame_number;
    }

    delete[] p_normal_data;
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