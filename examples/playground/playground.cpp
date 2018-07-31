// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

// RealSense SDK
#include <librealsense2/rs.hpp>
#include <librealsense2/hpp/rs_internal.hpp>

// RealSense SDK helpers
#include "../example.hpp"
#include <imgui.h>
#include "imgui_impl_glfw.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Standard
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <future>
#include <string>

// Module include
#include "json.hpp"
using json = nlohmann::json;

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

/**
Helper class for controlling the filter's GUI element
*/
struct filter_slider_ui
{
    std::string name;
    std::string label;
    std::string description;
    bool is_int;
    float value;
    rs2::option_range range;

    // bool render(const float3& location, bool enabled);
    static bool is_all_integers(const rs2::option_range& range);
};

/**
Class to encapsulate a filter alongside its options
*/
class filter_options
{
public:
    filter_options(const std::string name, rs2::process_interface& filter);
    filter_options(filter_options&& other);
    std::string filter_name;                                   // Friendly name of the filter
    rs2::process_interface& filter;                            // The filter in use
    std::map<rs2_option, filter_slider_ui> supported_options;  // maps from an option supported by the filter, to the corresponding slider
    std::atomic_bool is_enabled;                               // A boolean controlled by the user that determines whether to apply the filter or not
};

/**
Constructor for filter_options, takes a name and a filter.
*/
filter_options::filter_options(const std::string name, rs2::process_interface& filter) :
    filter_name(name),
    filter(filter),
    is_enabled(true)
{
    const std::array<rs2_option, 3> possible_filter_options = {
        RS2_OPTION_FILTER_MAGNITUDE,
        RS2_OPTION_FILTER_SMOOTH_ALPHA,
        RS2_OPTION_FILTER_SMOOTH_DELTA
    };

    //Go over each filter option and create a slider for it
    for (rs2_option opt : possible_filter_options)
    {
        if (filter.supports(opt))
        {
            rs2::option_range range = filter.get_option_range(opt);
            supported_options[opt].range = range;
            supported_options[opt].value = range.def; // default value
            supported_options[opt].is_int = filter_slider_ui::is_all_integers(range);
            supported_options[opt].description = filter.get_option_description(opt);
            std::string opt_name = rs2_option_to_string(opt);
            supported_options[opt].name = name + "_" + opt_name;
            std::string prefix = "Filter ";
            supported_options[opt].label = name + " " + opt_name.substr(prefix.length());
        }
    }
}

filter_options::filter_options(filter_options&& other) :
    filter_name(std::move(other.filter_name)),
    filter(other.filter),
    supported_options(std::move(other.supported_options)),
    is_enabled(other.is_enabled.load())
{
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

void dump_data(rs2::video_stream_profile& color_stream, rs2::video_stream_profile& depth_stream, float depth_scale)
{
    rs2_intrinsics color_intrinsic = color_stream.get_intrinsics();
    rs2_intrinsics depth_intrinsic = depth_stream.get_intrinsics();;
    rs2_extrinsics depth_to_color  = depth_stream.get_extrinsics_to(color_stream);

    printf("=== camera profiles ===\n");
    printf("color intrin: wh[%dx%d] ppx[%f,%f] f[%f,%f] model[%d] coef[%f,%f,%f,%f,%f]\n",
        color_intrinsic.width, color_intrinsic.height,
        color_intrinsic.ppx, color_intrinsic.ppy,
        color_intrinsic.fx, color_intrinsic.fy,
        color_intrinsic.model,
        color_intrinsic.coeffs[0],color_intrinsic.coeffs[1],color_intrinsic.coeffs[2],color_intrinsic.coeffs[3],color_intrinsic.coeffs[4]
    );
    printf("depth intrin: wh[%dx%d] ppx[%f,%f] f[%f,%f] model[%d] coef[%f,%f,%f,%f,%f]\n",
        depth_intrinsic.width, depth_intrinsic.height,
        depth_intrinsic.ppx, depth_intrinsic.ppy,
        depth_intrinsic.fx, depth_intrinsic.fy,
        depth_intrinsic.model,
        depth_intrinsic.coeffs[0],depth_intrinsic.coeffs[1],depth_intrinsic.coeffs[2],depth_intrinsic.coeffs[3],depth_intrinsic.coeffs[4]
    );

    printf("depth scale=[%f]\n", depth_scale);
    for(int i=0 ; i<9 ; ++i){
        printf("depth to color: rot[%d]=[%f]\n", i, depth_to_color.rotation[i]);
    }
    for(int i=0 ; i<3 ; ++i){
        printf("depth to color: trans[%d]=[%f]\n", i, depth_to_color.translation[i]);
    }


    json camera_profile;

    auto make_json_intrin = [&camera_profile](rs2_intrinsics& intinc, const char* camera_name)
    {
        camera_profile[camera_name]["width"] = intinc.width;
        camera_profile[camera_name]["height"] = intinc.height;
        camera_profile[camera_name]["ppx"] = intinc.ppx;
        camera_profile[camera_name]["ppy"] = intinc.ppy;
        camera_profile[camera_name]["fx"] = intinc.fx;
        camera_profile[camera_name]["fy"] = intinc.fy;
        camera_profile[camera_name]["distortion"] = intinc.model;
        camera_profile[camera_name]["coeff"] = intinc.coeffs;
    };

    make_json_intrin(color_intrinsic, "color");
    make_json_intrin(depth_intrinsic, "depth");

    auto make_json_extrin = [&camera_profile](rs2_extrinsics& extrinc, const char* camera_name)
    {
        camera_profile[camera_name]["rotation"] = extrinc.rotation;
        camera_profile[camera_name]["translation"] = extrinc.translation;
    };

    make_json_extrin(depth_to_color, "depth_to_color");

    camera_profile["depth_scale"] = depth_scale;

    char filename[128];
    snprintf(filename, sizeof(filename), "camera_profile.json");
    std::ofstream o(filename);
    o << std::setw(4) << camera_profile << std::endl;
    o.close();

    std::ifstream i(filename);
    json camera_profile_read;
    i >> camera_profile_read;
    i.close();

    // printf("input_intrinsic wh[%dx%d] ppx[%f,%f] coeff[%f,%f,%f] rot[%f,%f,%f] trans[%f,%f,%f]\n",
    //     (int)camera_profile_read["color"]["width"], (int)camera_profile_read["color"]["height"],
    //     (float)camera_profile_read["color"]["ppx"], (float)camera_profile_read["color"]["ppy"],
    //     (float)camera_profile_read["color"]["coeff"][0], (float)camera_profile_read["color"]["coeff"][1], (float)camera_profile_read["color"]["coeff"][2],
    //     (float)camera_profile_read["depth_to_color"]["rotation"][0],(float)camera_profile_read["depth_to_color"]["rotation"][1],(float)camera_profile_read["depth_to_color"]["rotation"][2],
    //     (float)camera_profile_read["depth_to_color"]["translation"][0],(float)camera_profile_read["depth_to_color"]["translation"][1],(float)camera_profile_read["depth_to_color"]["translation"][2]
    // ); 
}

void dump_depth_map(const rs2::depth_frame& depth_frame, const char* ext_name)
{
    printf("dump depthmap +\n");

    const uint16_t* p_depth_frame = reinterpret_cast<const uint16_t*>(depth_frame.get_data());

    int width = depth_frame.get_width();
    int height = depth_frame.get_height();

    char filename[128];
    using namespace std::chrono;
    milliseconds timestamp = duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()
    );
    snprintf(filename, sizeof(filename), "depthmap_%s_%d_%d_%d_%d.dat", ext_name, width, height, sizeof(uint16_t), timestamp.count());
    printf("%s\n", filename);

    std::ofstream out(filename, std::ios::out | std::ios::binary);

    if( !out )
    {
        printf("[E] Cannot open file!\n");
        return;
    }

    out.write((char*)p_depth_frame, width*height*sizeof(uint16_t));
    out.close();

    // check
    uint16_t* depth_read = new uint16_t[width*height];
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    in.read((char*)depth_read, width*height*sizeof(uint16_t));
    for(int i=0 ; i<width*height ; ++i)
    {
        if( depth_read[i] != p_depth_frame[i] )
        {
            printf("[E] check file failed!\n");
        }
    }
    printf("check file passed\n");

    delete[] depth_read;

    printf("dump depthmap -\n");
}

std::vector<uint8_t*> tmp_buffer_depth;
std::vector<uint8_t*> tmp_buffer_color;

std::string dump_buffer_to_memory(
    const rs2::video_frame& frame,
    const char* type,
    const char* extension,
    const std::chrono::milliseconds& timestamp,
    const int& index,
    const int& width,
    const int& height
){
    const uint16_t* p_frame = reinterpret_cast<const uint16_t*>(frame.get_data());
    const uint8_t* p_frame_byte = reinterpret_cast<const uint8_t*>(frame.get_data());

    char filename[128];
    snprintf(filename, sizeof(filename), "dump/%d_%s_%dx%d_ts[%lld].%s", 
        index, type, width, height, timestamp.count(), extension
    );    
    printf("%s\n", filename);
    std::string ret(filename);

    if( strcmp(extension,"png") == 0 ){
        // PNG
        rs2::colorizer colorer;
        rs2::video_frame vf = colorer(frame);
        // stbi_write_png(
        //     filename,
        //     vf.get_width(), vf.get_height(),
        //     vf.get_bytes_per_pixel(), vf.get_data(), vf.get_stride_in_bytes()
        // );
        std::memcpy(tmp_buffer_color[index], p_frame_byte, width*height*vf.get_bytes_per_pixel());
    }else{
        // uint16
        // std::ofstream out(filename, std::ios::out | std::ios::binary);
        // if( !out ){
        //     printf("[E] Cannot open file!\n");
        //     return ret;
        // }
        // out.write((char*)p_frame, width*height*sizeof(uint16_t));
        // out.close();
        std::memcpy(tmp_buffer_depth[index], p_frame_byte, width*height*sizeof(uint16_t));
    }
    return ret;
}

std::string dump_from_memory(
    const char* type,
    const char* extension,
    const uint64_t& timestamp,
    const int& index,
    const int& width,
    const int& height,
    const int& bpp
){
    char filename[128];
    snprintf(filename, sizeof(filename), "dump/%d_%s_%dx%d_ts[%lld].%s", 
        index, type, width, height, timestamp, extension
    );    
    printf("%s\n", filename);
    std::string ret(filename);

    if( strcmp(extension,"png") == 0 ){
        // PNG
        void* p_frame = static_cast<void*>(tmp_buffer_color[index]);
        stbi_write_png(
            filename,
            width, height,
            bpp, p_frame, width*bpp
        );
    }else{
        // uint16
        uint8_t* p_frame = tmp_buffer_depth[index];
        std::ofstream out(filename, std::ios::out | std::ios::binary);
        if( !out ){
            printf("[E] Cannot open file!\n");
            return ret;
        }
        out.write((char*)p_frame, width*height*bpp);
        out.close();
    }
    return ret;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                     These parameters are reconfigurable                                        //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define STREAM          RS2_STREAM_COLOR  // rs2_stream is a types of data provided by RealSense device           //
#define FORMAT          RS2_FORMAT_RGB8   // rs2_format is identifies how binary data is encoded within a frame   //
#define WIDTH           1920              // Defines the number of columns for each frame                         //
#define HEIGHT          1080              // Defines the number of lines for each frame                           //
#define FPS             30                // Defines the rate of frames per second                                //
#define STREAM_INDEX    0                 // Defines the stream index, used for multiple streams of the same type //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define STREAM_D        RS2_STREAM_DEPTH  // rs2_stream is a types of data provided by RealSense device           //
#define FORMAT_D        RS2_FORMAT_Z16    // rs2_format is identifies how binary data is encoded within a frame   //
#define WIDTH_D         640               // Defines the number of columns for each frame                         //
#define HEIGHT_D        480               // Defines the number of lines for each frame                           //
#define FPS_D           30                // Defines the rate of frames per second                                //
#define STREAM_INDEX_D  0                 // Defines the stream index, used for multiple streams of the same type //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int dump_sequence_data(const int& frame_dump_start, const int& frame_dump_end)
{
    printf("dump_sequence_data\n");
    // Parameters
    int W_win = 1280;
    int H_win = 720;

    // UI init
    window app(W_win, H_win, "Dump Sequence Data"); // Simple window handling
    ImGui_ImplGlfw_Init(app, false);           // ImGui library intializition
    texture renderer;                          // Helper for renderig images   

    // Device init
    rs2::pipeline pipe;
    rs2::config my_config;
    my_config.enable_stream(STREAM, STREAM_INDEX, WIDTH, HEIGHT, FORMAT, FPS);
    my_config.enable_stream(STREAM_D, STREAM_INDEX_D, WIDTH_D, HEIGHT_D, FORMAT_D, FPS_D);
    rs2::pipeline_profile profile = pipe.start(my_config);

    rs2::stream_profile color_stream_profile = get_color_stream_profile(profile.get_streams());
    rs2::video_stream_profile color_video_stream_profile = color_stream_profile.as<rs2::video_stream_profile>();
    
    rs2::stream_profile depth_stream_profile = get_depth_stream_profile(profile.get_streams());
    rs2::video_stream_profile depth_video_stream_profile = depth_stream_profile.as<rs2::video_stream_profile>();
    float depth_scale = get_depth_scale(profile.get_device());
    dump_data(color_video_stream_profile, depth_video_stream_profile, depth_scale);

    // Other init
    rs2::colorizer colorer;
    colorer.set_option(RS2_OPTION_COLOR_SCHEME, 2.0f); // grayscale
    rs2::align aligner(RS2_STREAM_COLOR); // Depth align to Color

    int frame_number = 0;
    json dump_list;
    int dump_count = frame_dump_end-frame_dump_start;
    dump_list["dump_count"] = dump_count;
    for(int i=0 ; i< dump_count ; ++i){
        tmp_buffer_color.push_back(new uint8_t[WIDTH*HEIGHT*3]);     // BPP == 3
        tmp_buffer_depth.push_back(new uint8_t[WIDTH_D*HEIGHT_D*2]); // BPP == 2
    }

    while (app) // Application still alive?
    {
        // printf("frame:[%d]\n", frame_number);
        using namespace std::chrono;
        milliseconds timestamp = duration_cast<milliseconds>(
            system_clock::now().time_since_epoch()
        );
        // raw data
        rs2::frameset frameset = pipe.wait_for_frames();
        rs2::video_frame color_frame = frameset.get_color_frame();
        rs2::video_frame raw_depth_frame = frameset.get_depth_frame();
        if( !color_frame || !raw_depth_frame ){
            continue;
        }

        float w = static_cast<float>(app.width());
        float h = static_cast<float>(app.height());
        
        int   rowCount  = 1;
        int   colCount  = 2;
        float showOffset[1] = {0.0};
        auto addShow = [&](rs2::video_frame* _vFrame, int row, const char* _str, bool _isDepth = false)
        {
            rect frame_rect{showOffset[row], (h/rowCount)*row, w/colCount, h/rowCount};
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
            showOffset[row] += frame_rect.w;
        };

        // row 0
        addShow(&color_frame,      0, "Color");
        addShow(&raw_depth_frame,   0, "Depth Map", true);

        if( frame_number >= frame_dump_start && frame_number < frame_dump_end ){
            int dump_index = frame_number-frame_dump_start;
            //std::string filename_depth = dump_buffer(raw_depth_frame, "depth", "dat", timestamp, frame_number, raw_depth_frame.get_width(), raw_depth_frame.get_height());            
            std::future<std::string> task_dump_depth = std::async(
                std::launch::async, 
                [&]{
                    dump_list["depth"][dump_index]["type"] = "depth";
                    dump_list["depth"][dump_index]["extension"] = "dat";
                    dump_list["depth"][dump_index]["timestamp"] = (uint64_t)timestamp.count();
                    dump_list["depth"][dump_index]["index"] = (uint64_t)dump_index;
                    dump_list["depth"][dump_index]["width"] = raw_depth_frame.get_width();
                    dump_list["depth"][dump_index]["height"] = raw_depth_frame.get_height();
                    dump_list["depth"][dump_index]["bpp"] = raw_depth_frame.get_bytes_per_pixel();
                    return dump_buffer_to_memory(raw_depth_frame, "depth", "dat", timestamp, dump_index, raw_depth_frame.get_width(), raw_depth_frame.get_height());
                }
            );

            dump_list["color"][dump_index]["type"] = "color";
            dump_list["color"][dump_index]["extension"] = "png";
            dump_list["color"][dump_index]["timestamp"] = (uint64_t)timestamp.count();
            dump_list["color"][dump_index]["index"] = (uint64_t)dump_index;
            dump_list["color"][dump_index]["width"] = color_frame.get_width();
            dump_list["color"][dump_index]["height"] = color_frame.get_height();
            dump_list["color"][dump_index]["bpp"] = color_frame.get_bytes_per_pixel();
            std::string filename_color = dump_buffer_to_memory(color_frame, "color", "png", timestamp, dump_index, color_frame.get_width(), color_frame.get_height());

            dump_list["color"][dump_index]["filename"] = filename_color;
            dump_list["depth"][dump_index]["filename"] = task_dump_depth.get();
        }

        ++frame_number;
    }

    // dump from memory
    {
        printf("dump from memory +\n");
        for(int i=0 ; i<dump_count ; ++i){
            dump_from_memory(
                "color",
                "png",
                dump_list["color"][i]["timestamp"],
                dump_list["color"][i]["index"],
                dump_list["color"][i]["width"],
                dump_list["color"][i]["height"],
                dump_list["color"][i]["bpp"]
            );
            dump_from_memory(
                "depth",
                "dat",
                dump_list["depth"][i]["timestamp"],
                dump_list["depth"][i]["index"],
                dump_list["depth"][i]["width"],
                dump_list["depth"][i]["height"],
                dump_list["depth"][i]["bpp"]
            );
        }
        printf("dump from memory -\n");
    }

    char filename[128];
    snprintf(filename, sizeof(filename), "dump/dump_list.json");
    std::ofstream o(filename);
    o << std::setw(4) << dump_list << std::endl;
    o.close();

    // release working buffer
    for(int i=0 ; i<dump_count ; ++i){
        delete[] tmp_buffer_color[i];
        delete[] tmp_buffer_depth[i];
    }
    return EXIT_SUCCESS;
}

void dump_default_filter_parameters(std::vector<filter_options>& filters, const char* filename)
{
    // dump default filer parameters
    json filter_options_default_json;

    for (auto& filter : filters){
        // printf("[%s] parameters:", filter.filter_name.c_str());
        for (auto& option_slider_pair : filter.supported_options){
            filter_slider_ui& slider = option_slider_pair.second;
            if( filter.is_enabled ){
                filter_options_default_json[filter.filter_name][slider.name] = slider.value;
                float range[2] = {slider.range.min, slider.range.max};
                std::string range_name = slider.name + " Range";
                filter_options_default_json[filter.filter_name][range_name] = range; 
                // printf(" [%s](%f) [%f-%f]", slider.name.c_str(), slider.value,  slider.range.min, slider.range.max);
            }
        }
        printf("\n");
    }

    std::ofstream o(filename);
    o << std::setw(4) << filter_options_default_json << std::endl;
    o.close();
}

void update_filter_parameters(std::vector<filter_options>& filters, json& filter_options_json)
{
    system("cls");
    for (auto& filter : filters){
        printf("[%s] parameters:", filter.filter_name.c_str());
        for (auto& option_slider_pair : filter.supported_options){
            filter_slider_ui& slider = option_slider_pair.second;
            if( filter.is_enabled ){
                slider.value = filter_options_json[filter.filter_name][slider.name];
                printf(" [%s](%f) ", slider.name.c_str(), slider.value);
                filter.filter.set_option(option_slider_pair.first, slider.value);
            }
        }
        printf("\n");
    }
}

int filter_test()
{
    // Parameters
    int W_win = 1280;
    int H_win = 720;
    float depth_clipping_distance = 1.0f;
    int BPP_Depth = 2;


    // UI init
    window app(W_win, H_win, "My playground"); // Simple window handling
    ImGui_ImplGlfw_Init(app, false);           // ImGui library intializition
    texture renderer;                          // Helper for renderig images

    // Device init
    rs2::pipeline pipe;
    rs2::config my_config;
    my_config.enable_stream(STREAM, STREAM_INDEX, WIDTH, HEIGHT, FORMAT, FPS);
    my_config.enable_stream(STREAM_D, STREAM_INDEX_D, WIDTH_D, HEIGHT_D, FORMAT_D, FPS_D);
    rs2::pipeline_profile profile = pipe.start(my_config);

    rs2::stream_profile color_stream_profile = get_color_stream_profile(profile.get_streams());
    rs2::video_stream_profile color_video_stream_profile = color_stream_profile.as<rs2::video_stream_profile>();
    
    rs2::stream_profile depth_stream_profile = get_depth_stream_profile(profile.get_streams());
    rs2::video_stream_profile depth_video_stream_profile = depth_stream_profile.as<rs2::video_stream_profile>();
    float depth_scale = get_depth_scale(profile.get_device());

    // Create software-only device
    rs2::software_device dev;
    auto filtered_depth_sensor          = dev.add_sensor("FilteredDepth"); // Define single sensor
    uint8_t* p_filtered_depth = new uint8_t[WIDTH_D*HEIGHT_D*BPP_Depth];
    auto filtered_depth_stream = filtered_depth_sensor.add_video_stream({
        RS2_STREAM_DEPTH, 0, 0,
        WIDTH_D, HEIGHT_D, 60,
        BPP_Depth,
        RS2_FORMAT_DISPARITY16, depth_video_stream_profile.get_intrinsics() }
    );

    rs2::syncer sync;
    filtered_depth_sensor.open(filtered_depth_stream);
    filtered_depth_sensor.start(sync);

    // Other init
    rs2::colorizer colorer;
    colorer.set_option(RS2_OPTION_COLOR_SCHEME, 2.0f); // grayscale
    colorer.set_option(RS2_OPTION_HISTOGRAM_EQUALIZATION_ENABLED, 0.0f);

    // Filters
    rs2::spatial_filter spat_filter;    // Spatial    - edge-preserving spatial smoothing
    rs2::temporal_filter temp_filter;   // Temporal   - reduces temporal noise
    // Initialize a vector that holds filters and their options
    std::vector<filter_options> filters;
    filters.emplace_back("Spatial", spat_filter);
    filters.emplace_back("Temporal", temp_filter);

    char json_filename[128];
    snprintf(json_filename, sizeof(json_filename), "filter_options.json");
    dump_default_filter_parameters(filters, json_filename);

    int frame_number = 0;
    while (app) // Application still alive?
    {
        // printf("frame:[%d]\n", frame_number);
        using namespace std::chrono;
        milliseconds timestamp = duration_cast<milliseconds>(
            system_clock::now().time_since_epoch()
        );
        // raw data
        rs2::frameset frameset = pipe.wait_for_frames();
        rs2::video_frame color_frame = frameset.get_color_frame();
        rs2::video_frame raw_depth_frame = frameset.get_depth_frame();
        if ( !color_frame || !raw_depth_frame )
        {
            continue;
        }

        const uint8_t* p_raw_depth_frame = static_cast<const uint8_t*>(raw_depth_frame.get_data());
        update_sw_sensor(filtered_depth_sensor, depth_stream_profile, frame_number, p_raw_depth_frame, 
            WIDTH_D, HEIGHT_D, BPP_Depth
        );

        rs2::frameset frameset_sw            = sync.wait_for_frames();
        rs2::video_frame clone_depth_frame   = frameset_sw.get_depth_frame();
        // printf("frameset_sw size(%d)\n", frameset_sw.size());
        
        // printf("frameset_sw_aligned size(%d)\n", frameset_sw_aligned.size());
        if( !clone_depth_frame )
        {
            continue;
        }

        // filers
        if( frame_number % 30 == 0 ){
            // read from json
            std::ifstream input(json_filename);
            json filter_options_json;
            input >> filter_options_json;
            input.close();
            update_filter_parameters(filters, filter_options_json);
        }
        for (auto&& filter : filters){
            clone_depth_frame = filter.filter.process(clone_depth_frame);
        }
        
        // render
        float w = static_cast<float>(app.width());
        float h = static_cast<float>(app.height());

        int   rowCount  = 2;
        int   colCount  = 2;
        float showOffset[2] = {0.0, 0.0};
        auto addShow = [&](rs2::video_frame* _vFrame, int row, const char* _str, bool _isDepth = false)
        {
            rect frame_rect{showOffset[row], (h/rowCount)*row, w/colCount, h/rowCount};
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
            showOffset[row] += frame_rect.w;
        };

        // row 0
        addShow(&color_frame,       0, "RGB");
        
        // row 1
        addShow(&raw_depth_frame,     1, "Depth Map", true);
        addShow(&clone_depth_frame,   1, "Depth Map Filtered", true);

        // printf("frame:[%d] duration[%d](ms)\n",
        //     frame_number,
        //     duration_cast<milliseconds>(system_clock::now().time_since_epoch()) - timestamp
        // );

        ++frame_number;
    }

    return EXIT_SUCCESS;
}

int show_everything()
{
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
    rs2::config my_config;
    my_config.enable_stream(STREAM, STREAM_INDEX, WIDTH, HEIGHT, FORMAT, FPS);
    my_config.enable_stream(STREAM_D, STREAM_INDEX_D, WIDTH_D, HEIGHT_D, FORMAT_D, FPS_D);
    rs2::pipeline_profile profile = pipe.start(my_config);

    rs2::stream_profile color_stream_profile = get_color_stream_profile(profile.get_streams());
    rs2::video_stream_profile color_video_stream_profile = color_stream_profile.as<rs2::video_stream_profile>();
    
    rs2::stream_profile depth_stream_profile = get_depth_stream_profile(profile.get_streams());
    rs2::video_stream_profile depth_video_stream_profile = depth_stream_profile.as<rs2::video_stream_profile>();
    float depth_scale = get_depth_scale(profile.get_device());

    dump_data(color_video_stream_profile, depth_video_stream_profile, depth_scale);
  
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

    rs2::syncer sync;
    normal_map_sensor.open(normal_map_stream);
    normal_map_sensor.start(sync);
    
    rs2::syncer sync_aligned;
    normal_map_sensor_aligned.open(normal_map_stream_aligned);    
    normal_map_sensor_aligned.start(sync_aligned);

    printf("color[%dx%d], depth[%dx%d], normal[%dx%d]\n",
        color_video_stream_profile.width(), color_video_stream_profile.height(),
        depth_video_stream_profile.width(), depth_video_stream_profile.height(),
        W_normal, H_normal
    );

    // Other init
    rs2::colorizer colorer;
    colorer.set_option(RS2_OPTION_COLOR_SCHEME, 2.0f); // grayscale
    /*
    color_map->set_description(0.f, "Jet");
    color_map->set_description(1.f, "Classic");
    color_map->set_description(2.f, "White to Black");
    color_map->set_description(3.f, "Black to White");
    color_map->set_description(4.f, "Bio");
    color_map->set_description(5.f, "Cold");
    color_map->set_description(6.f, "Warm");
    color_map->set_description(7.f, "Quantized");
    color_map->set_description(8.f, "Pattern");
    */
    // Disable histogram equalization
    // colorer.set_option(RS2_OPTION_HISTOGRAM_EQUALIZATION_ENABLED, 0.0f);

    rs2::align aligner(RS2_STREAM_COLOR); // Depth align to Color

    int frame_number = 0;
    while (app) // Application still alive?
    {
        // printf("frame:[%d]\n", frame_number);
        using namespace std::chrono;
        milliseconds timestamp = duration_cast<milliseconds>(
            system_clock::now().time_since_epoch()
        );
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
        if(1)
        {
            make_normal_map(
                p_aligned_normal_data, W_aligned_normal, H_aligned_normal, BPP_normal,
                aligned_depth_frame, depth_scale, depth_clipping_distance
            );
        }
        update_sw_sensor(normal_map_sensor_aligned, normal_map_stream_aligned, frame_number, p_aligned_normal_data, W_aligned_normal, H_aligned_normal, BPP_normal);

        // normal map from raw data
        if(1)
        {
            make_normal_map(
                p_normal_data, W_normal, H_normal, BPP_normal,
                raw_depth_frame, depth_scale, depth_clipping_distance
            );
        }
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
        
        int   rowCount  = 2;
        int   colCount  = 3;
        float showOffset[2] = {0.0, 0.0};
        auto addShow = [&](rs2::video_frame* _vFrame, int row, const char* _str, bool _isDepth = false)
        {
            rect frame_rect{showOffset[row], (h/rowCount)*row, w/colCount, h/rowCount};
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
            showOffset[row] += frame_rect.w;
        };

        // row 0
        addShow(&normal_frame,      0, "Normal Map");
        addShow(&raw_depth_frame,   0, "Depth Map", true);

        // row 1
        addShow(static_cast<rs2::video_frame*>(&aligned_normal_frame),  1, "Aligned Normal Map");
        addShow(static_cast<rs2::video_frame*>(&aligned_depth_frame),   1, "Aligned Depth Map", true);
        addShow(&color_frame,       1, "RGB");

        printf("frame:[%d] duration[%d](ms)\n",
            frame_number,
            duration_cast<milliseconds>(system_clock::now().time_since_epoch()) - timestamp
        );

        // if( frame_number == 10 ){
        //     dump_depth_map(raw_depth_frame, "raw");
        //     dump_depth_map(aligned_depth_frame, "aligned");
        //     break;
        // }

        ++frame_number;
    }

    delete[] p_normal_data;
    return EXIT_SUCCESS;
}

int main(int argc, char * argv[]) try
{
    printf("My Playground\n");

    // int ret = show_everything();
    // int ret = dump_sequence_data(120, 130);
    int ret = filter_test();
    
    system("pause");
    return ret;
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

/**
  Helper function for deciding on int ot float slider
*/
bool filter_slider_ui::is_all_integers(const rs2::option_range& range)
{
    const auto is_integer = [](float f)
    {
        return (fabs(fmod(f, 1)) < std::numeric_limits<float>::min());
    };

    return is_integer(range.min) && is_integer(range.max) &&
        is_integer(range.def) && is_integer(range.step);
}