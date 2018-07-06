// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp>
#include "../example.hpp"
#include <imgui.h>
#include "imgui_impl_glfw.h"

#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <chrono>

// #define SIMPLE_TEST
#define MAX_LOOP_COUNT 20

void render_slider(rect location, float& clipping_dist);
void remove_background(rs2::video_frame& other, const rs2::depth_frame& depth_frame, float depth_scale, float clipping_dist);
void draw_original_depthmap(rs2::video_frame& other, const rs2::depth_frame& depth_frame, float depth_scale, float clipping_dist);
float get_depth_scale(rs2::device dev);
rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams);
bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev);
void dump_depth_map(const rs2::depth_frame& depth_frame, float depth_scale);
void find_camera_params(const std::vector<rs2::stream_profile>& streams);

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

int main(int argc, char * argv[]) try
{
    // Create and initialize GUI related objects
    window app(1280, 720, "CPP - Align Example"); // Simple window handling
    ImGui_ImplGlfw_Init(app, false);      // ImGui library intializition
    rs2::colorizer colorer;                     // Helper to colorize depth images
    texture renderer;                     // Helper for renderig images

    // Create a pipeline to easily configure and start the camera
    rs2::pipeline pipe;
    //Calling pipeline's start() without any additional parameters will start the first device
    // with its default streams.
    //The start function returns the pipeline profile which the pipeline used to start the device
    rs2::pipeline_profile profile = pipe.start();

    // Each depth camera might have different units for depth pixels, so we get it here
    // Using the pipeline's profile, we can retrieve the device that the pipeline uses
    float depth_scale = get_depth_scale(profile.get_device());

    //Pipeline could choose a device that does not have a color stream
    //If there is no color stream, choose to align depth to another stream
    rs2_stream align_to = find_stream_to_align(profile.get_streams());

    // Create a rs2::align object.
    // rs2::align allows us to perform alignment of depth frames to others frames
    //The "align_to" is the stream type to which we plan to align depth frames.
    rs2::align align(align_to);

    // Define a variable for controlling the distance to clip
    float depth_clipping_distance = 1.0f;

    int loop_count = 0;

    find_camera_params(profile.get_streams());

    while (app) // Application still alive?
    {
        // Using the align object, we block the application until a frameset is available
        rs2::frameset frameset = pipe.wait_for_frames();

        // rs2::pipeline::wait_for_frames() can replace the device it uses in case of device error or disconnection.
        // Since rs2::align is aligning depth to some other stream, we need to make sure that the stream was not changed
        //  after the call to wait_for_frames();
        if (profile_changed(pipe.get_active_profile().get_streams(), profile.get_streams()))
        {
            //If the profile was changed, update the align object, and also get the new device's depth scale
            profile = pipe.get_active_profile();
            align_to = find_stream_to_align(profile.get_streams());
            align = rs2::align(align_to);
            depth_scale = get_depth_scale(profile.get_device());
        }

        //Get processed aligned frame
        auto processed = align.process(frameset);

        // Trying to get both other and aligned depth frames
        rs2::video_frame other_frame = processed.first(align_to);
        rs2::depth_frame aligned_depth_frame = processed.get_depth_frame();

        //If one of them is unavailable, continue iteration
        if (!aligned_depth_frame || !other_frame)
        {
            continue;
        }

        // Passing both frames to remove_background so it will "strip" the background
        // NOTE: in this example, we alter the buffer of the other frame, instead of copying it and altering the copy
        //       This behavior is not recommended in real application since the other frame could be used elsewhere
        // remove_background(other_frame, aligned_depth_frame, depth_scale, depth_clipping_distance);
        draw_original_depthmap(other_frame, frameset.get_depth_frame(), depth_scale, depth_clipping_distance);

        // Taking dimensions of the window for rendering purposes
        float w = static_cast<float>(app.width());
        float h = static_cast<float>(app.height());

        // At this point, "other_frame" is an altered frame, stripped form its background
        // Calculating the position to place the frame in the window
        rect altered_other_frame_rect{ 0, 0, w, h };
        altered_other_frame_rect = altered_other_frame_rect.adjust_ratio({ static_cast<float>(other_frame.get_width()),static_cast<float>(other_frame.get_height()) });

        // Render aligned image
        renderer.render(other_frame, altered_other_frame_rect);

        // The example also renders the depth frame, as a picture-in-picture
        // Calculating the position to place the depth frame in the window
        rect pip_stream{ 0, 0, w / 3, h / 3 };
        pip_stream = pip_stream.adjust_ratio({ static_cast<float>(aligned_depth_frame.get_width()),static_cast<float>(aligned_depth_frame.get_height()) });
        pip_stream.x = altered_other_frame_rect.x + altered_other_frame_rect.w - pip_stream.w + 100;
        pip_stream.y = altered_other_frame_rect.y + altered_other_frame_rect.h - pip_stream.h;

        // Render depth (as picture in pipcture)
        renderer.upload(colorer(aligned_depth_frame));
        renderer.show(pip_stream);

        // Using ImGui library to provide a slide controller to select the depth clipping distance
        ImGui_ImplGlfw_NewFrame(1);
        render_slider({ 5.f, 0, w, h }, depth_clipping_distance);
        ImGui::Render();
#if 0
        printf("loop(%d/%d)\n", loop_count, MAX_LOOP_COUNT);
        if( loop_count == MAX_LOOP_COUNT )
        {
            dump_depth_map(aligned_depth_frame, depth_scale);
            break;
        }
        ++loop_count;
#endif
    }
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

void render_slider(rect location, float& clipping_dist)
{
    // Some trickery to display the control nicely
    static const int flags = ImGuiWindowFlags_NoCollapse
        | ImGuiWindowFlags_NoScrollbar
        | ImGuiWindowFlags_NoSavedSettings
        | ImGuiWindowFlags_NoTitleBar
        | ImGuiWindowFlags_NoResize
        | ImGuiWindowFlags_NoMove;
    const int pixels_to_buttom_of_stream_text = 25;
    const float slider_window_width = 30;

    ImGui::SetNextWindowPos({ location.x, location.y + pixels_to_buttom_of_stream_text });
    ImGui::SetNextWindowSize({ slider_window_width + 20, location.h - (pixels_to_buttom_of_stream_text * 2) });

    //Render the vertical slider
    ImGui::Begin("slider", nullptr, flags);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImColor(215.f / 255, 215.0f / 255, 215.0f / 255));
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, ImColor(215.f / 255, 215.0f / 255, 215.0f / 255));
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, ImColor(215.f / 255, 215.0f / 255, 215.0f / 255));
    auto slider_size = ImVec2(slider_window_width / 2, location.h - (pixels_to_buttom_of_stream_text * 2) - 20);
    ImGui::VSliderFloat("", slider_size, &clipping_dist, 0.0f, 6.0f, "", 1.0f, true);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Depth Clipping Distance: %.3f", clipping_dist);
    ImGui::PopStyleColor(3);

    //Display bars next to slider
    float bars_dist = (slider_size.y / 6.0f);
    for (int i = 0; i <= 6; i++)
    {
        ImGui::SetCursorPos({ slider_size.x, i * bars_dist });
        std::string bar_text = "- " + std::to_string(6-i) + "m";
        ImGui::Text("%s", bar_text.c_str());
    }
    ImGui::End();
}

void remove_background(rs2::video_frame& other_frame, const rs2::depth_frame& depth_frame, float depth_scale, float clipping_dist)
{
    const uint16_t* p_depth_frame = reinterpret_cast<const uint16_t*>(depth_frame.get_data());
    uint8_t* p_other_frame = reinterpret_cast<uint8_t*>(const_cast<void*>(other_frame.get_data()));

    int width = other_frame.get_width();
    int height = other_frame.get_height();
    int other_bpp = other_frame.get_bytes_per_pixel();

    auto get_pixel_depth = [&](const int& _x, const int& _y, const float& gain = 1.0f)
	{
		return depth_scale * p_depth_frame[(_y * width) + _x] * gain;
	};

    static int range = 255;

#ifdef SIMPLE_TEST
	Vector3 t(100, 99, 0.9);
	Vector3 l(99, 100, 0.9);
	Vector3 c(100, 100, 1.0);
	Vector3 normal = (l - c).crossProduct(t - c);
	printf("original   [%f,%f,%f] length(%f)\n", normal.x, normal.y, normal.z, normal.length());

	normal.normalize();
	printf("normalized [%f,%f,%f] length(%f)\n", normal.x, normal.y, normal.z, normal.length());

	normal.remap();
	printf("remapped   [%f,%f,%f] length(%f)\n", normal.x, normal.y, normal.z, normal.length());

	Vector3 rgb(normal.x*range, normal.y*range, normal.z*range);
	printf("rgb   [%f,%f,%f]\n", rgb.x, rgb.y, rgb.z);
#endif

    #pragma omp parallel for schedule(dynamic) //Using OpenMP to try to parallelise the loop
    for (int y = 1; y < height; y++)
    {
        auto depth_pixel_index = y * width;
        for (int x = 1; x < width; x++, ++depth_pixel_index)
        {
            // Get the depth value of the current pixel
            auto pixels_distance = depth_scale * p_depth_frame[depth_pixel_index];
            // Calculate the offset in other frame's buffer to current pixel
            auto offset = depth_pixel_index * other_bpp;

            // Check if the depth value is invalid (<=0) or greater than the threashold
            if (pixels_distance <= 0.0f || pixels_distance > clipping_dist)
            {
                // Set pixel to "background" color (0x00)
                std::memset(&p_other_frame[offset], 0x00, other_bpp);
            }
            else
            {
#ifndef SIMPLE_TEST
                // Amplify the depth value, ohterwise the depth variation of neighbor pixels are too small,
                // and the normal vector will be (0,0,1) for most cases.
                static float depth_gain = 1000.0f;
                Vector3 t(x,     y - 1, get_pixel_depth(x,     y - 1, depth_gain), width, height);
                Vector3 l(x - 1, y,     get_pixel_depth(x - 1, y,     depth_gain), width, height);
                Vector3 c(x,     y,     get_pixel_depth(x,     y,     depth_gain), width, height);
                Vector3 normal = (l - c).crossProduct(t - c);
                
                Vector3 rgb(0,0,0);
                if( false /*x == width/2 && y == height/2*/ )
                {
                    // print center pixel information
                    printf("[%d,%d] t[%f,%f,%f] l[%f,%f,%f] c[%f,%f,%f] depth_scale[%f]\n", x, y,
                        t.x, t.y, t.z,
                        l.x, l.y, l.z,
                        c.x, c.y, c.z,
                        depth_scale
                    );
                    printf("[%d,%d]original   [%f,%f,%f] length(%f)\n", x, y, normal.x, normal.y, normal.z, normal.length());
                    
                    normal.normalize();
                    printf("[%d,%d]normalize  [%f,%f,%f] length(%f)\n", x, y, normal.x, normal.y, normal.z, normal.length());
                    
                    normal.remap();
                    printf("[%d,%d]remapped   [%f,%f,%f] length(%f)\n", x, y, normal.x, normal.y, normal.z, normal.length());
                    
                    rgb = Vector3(normal.x*range, normal.y*range, normal.z*range);
                    printf("[%d,%d]rgb        [%f,%f,%f]\n", x, y, rgb.x, rgb.y, rgb.z);
                }
                else
                {
                    normal.normalize();
                    normal.remap();
                    rgb = Vector3(normal.x*range, normal.y*range, normal.z*range);
                }
#endif
                // remap 0.7~1.0 -> 0.0~1.0
                // rgb.x = rgb.y = rgb.z = ((get_pixel_depth(x, y)-0.7)/0.3)*255.0;

                // R, G, B channel:
                std::memset(&p_other_frame[offset], rgb.x, 1);
                std::memset(&p_other_frame[offset + 1], rgb.y, 1);
                std::memset(&p_other_frame[offset + 2], rgb.z, 1);
            }
        }
    }
}

void draw_original_depthmap(rs2::video_frame& other_frame, const rs2::depth_frame& depth_frame, float depth_scale, float clipping_dist)
{
    const uint16_t* p_depth_frame = reinterpret_cast<const uint16_t*>(depth_frame.get_data());
    uint8_t* p_other_frame = reinterpret_cast<uint8_t*>(const_cast<void*>(other_frame.get_data()));

    // clear color image
    int width = other_frame.get_width();
    int height = other_frame.get_height();
    int other_bpp = other_frame.get_bytes_per_pixel();
    std::memset(p_other_frame, 0, width*height*other_bpp);

    auto get_pixel_depth = [&](const int& _x, const int& _y, const float& gain = 1.0f)
	{
		return depth_scale * p_depth_frame[(_y * width) + _x] * gain;
	};

    static int range = 255;

    // draw normal map from original depthmap
    width = depth_frame.get_width();
    height = depth_frame.get_height();

    #pragma omp parallel for schedule(dynamic) //Using OpenMP to try to parallelise the loop
    for (int y = 1; y < height; y++)
    {
        for (int x = 1; x < width; x++)
        {
            // Get the depth value of the current pixel
            auto pixels_distance = get_pixel_depth(x,y);
            // Calculate the offset in other frame's buffer to current pixel
            auto offset = (y * width + x) * other_bpp;

            // Check if the depth value is invalid (<=0) or greater than the threashold
            if (pixels_distance <= 0.0f || pixels_distance > clipping_dist)
            {
                // Set pixel to "background" color (0x00)
                std::memset(&p_other_frame[offset], 0x00, other_bpp);
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
                std::memset(&p_other_frame[offset], rgb.x, 1);
                std::memset(&p_other_frame[offset + 1], rgb.y, 1);
                std::memset(&p_other_frame[offset + 2], rgb.z, 1);
            }
        }
    }
}

rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
{
    //Given a vector of streams, we try to find a depth stream and another stream to align depth with.
    //We prioritize color streams to make the view look better.
    //If color is not available, we take another stream that (other than depth)
    rs2_stream align_to = RS2_STREAM_ANY;
    bool depth_stream_found = false;
    bool color_stream_found = false;
    for (rs2::stream_profile sp : streams)
    {
        rs2_stream profile_stream = sp.stream_type();
        if (profile_stream != RS2_STREAM_DEPTH)
        {
            if (!color_stream_found)         //Prefer color
                align_to = profile_stream;

            if (profile_stream == RS2_STREAM_COLOR)
            {
                color_stream_found = true;
            }
        }
        else
        {
            depth_stream_found = true;
        }
    }

    if(!depth_stream_found)
        throw std::runtime_error("No Depth stream available");

    if (align_to == RS2_STREAM_ANY)
        throw std::runtime_error("No stream found to align with Depth");

    return align_to;
}

bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev)
{
    for (auto&& sp : prev)
    {
        //If previous profile is in current (maybe just added another)
        auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile& current_sp) { return sp.unique_id() == current_sp.unique_id(); });
        if (itr == std::end(current)) //If it previous stream wasn't found in current
        {
            return true;
        }
    }
    return false;
}

void dump_depth_map(const rs2::depth_frame& depth_frame, float depth_scale)
{
    printf("dump depthmap +\n");

    const uint16_t* p_depth_frame = reinterpret_cast<const uint16_t*>(depth_frame.get_data());

    int width = depth_frame.get_width();
    int height = depth_frame.get_height();

     printf("dump depthmap [%dx%d]\n", width, height);

    float* depth_in_meter = new float[width*height];
    int i = 0;
    for (int y = 0; y < height; y++)
    {
        auto depth_pixel_index = y * width;
        for (int x = 0; x < width; x++, ++depth_pixel_index)
        {
            depth_in_meter[i++] = depth_scale * p_depth_frame[depth_pixel_index];
        }
    }

    char filename[128];
    using namespace std::chrono;
    milliseconds timestamp = duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()
    );
    snprintf(filename, sizeof(filename), "depthmap_meter_%d_%d_%d_%d.dat", width, height, sizeof(float), timestamp.count());

    std::ofstream out(filename, std::ios::out | std::ios::binary);

    if( !out )
    {
        printf("[E] Cannot open file!\n");
        return;
    }

    out.write((char*)depth_in_meter, width*height*sizeof(float));
    out.close();

    // check
    float* depth_in_meter_read = new float[width*height];
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    in.read((char*)depth_in_meter_read, width*height*sizeof(float));
    for(int i=0 ; i<width*height ; ++i)
    {
        if( depth_in_meter_read[i] != depth_in_meter[i] )
        {
            printf("[E] check file failed!\n");
        }
    }
    printf("check file passed\n");

    delete[] depth_in_meter;

    printf("dump depthmap -\n");
}

void find_camera_params(const std::vector<rs2::stream_profile>& streams)
{
    for (rs2::stream_profile sp : streams)
    {
        rs2_stream profile_stream = sp.stream_type();
        if (profile_stream == RS2_STREAM_COLOR)
        {
            printf("find color stream\n");
            auto video_stream = sp.as<rs2::video_stream_profile>();
            //If the stream is indeed a video stream, we can now simply call get_intrinsics()
            rs2_intrinsics intrinsics = video_stream.get_intrinsics();

            auto principal_point = std::make_pair(intrinsics.ppx, intrinsics.ppy);
            auto focal_length = std::make_pair(intrinsics.fx, intrinsics.fy);
            rs2_distortion model = intrinsics.model;

            std::cout << "Principal Point         : " << principal_point.first << ", " << principal_point.second << std::endl;
            std::cout << "Focal Length            : " << focal_length.first << ", " << focal_length.second << std::endl;
            std::cout << "Distortion Model        : " << model << std::endl;
            std::cout << "Distortion Coefficients : [" << intrinsics.coeffs[0] << "," << intrinsics.coeffs[1] << "," <<
                intrinsics.coeffs[2] << "," << intrinsics.coeffs[3] << "," << intrinsics.coeffs[4] << "]" << std::endl;
        }
    }
}