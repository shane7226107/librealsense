// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include "../include/librealsense2/hpp/rs_sensor.hpp"
#include "../include/librealsense2/hpp/rs_processing.hpp"
#include "../include/librealsense2/rsutil.h"

#include "core/video.h"
#include "proc/synthetic-stream.h"
#include "environment.h"
#include "align.h"
#include "stream.h"

#define ALIGN_METHOD (2)

namespace librealsense
{
    template<class GET_DEPTH, class TRANSFER_PIXEL>
    void align_images(const rs2_intrinsics& depth_intrin, const rs2_extrinsics& depth_to_other,
        const rs2_intrinsics& other_intrin, GET_DEPTH get_depth, TRANSFER_PIXEL transfer_pixel)
    {
        // Iterate over the pixels of the depth image
#pragma omp parallel for schedule(dynamic)
        for (int depth_y = 0; depth_y < depth_intrin.height; ++depth_y)
        {
            int depth_pixel_index = depth_y * depth_intrin.width;
            for (int depth_x = 0; depth_x < depth_intrin.width; ++depth_x, ++depth_pixel_index)
            {
                // Skip over depth pixels with the value of zero, we have no depth data so we will not write anything into our aligned images
                if (float depth = get_depth(depth_pixel_index))
                {
                    // Map the top-left corner of the depth pixel onto the other image
                    float depth_pixel[2] = { depth_x - 0.5f, depth_y - 0.5f }, depth_point[3], other_point[3], other_pixel[2];
                    rs2_deproject_pixel_to_point(depth_point, &depth_intrin, depth_pixel, depth);
                    rs2_transform_point_to_point(other_point, &depth_to_other, depth_point);
                    rs2_project_point_to_pixel(other_pixel, &other_intrin, other_point);
                    const int other_x0 = static_cast<int>(other_pixel[0] + 0.5f); //rounding
                    const int other_y0 = static_cast<int>(other_pixel[1] + 0.5f);

                    // Map the bottom-right corner of the depth pixel onto the other image
                    depth_pixel[0] = depth_x + 0.5f; depth_pixel[1] = depth_y + 0.5f;
                    rs2_deproject_pixel_to_point(depth_point, &depth_intrin, depth_pixel, depth);
                    rs2_transform_point_to_point(other_point, &depth_to_other, depth_point);
                    rs2_project_point_to_pixel(other_pixel, &other_intrin, other_point);
                    const int other_x1 = static_cast<int>(other_pixel[0] + 0.5f);
                    const int other_y1 = static_cast<int>(other_pixel[1] + 0.5f);

                    if (other_x0 < 0 || other_y0 < 0 || other_x1 >= other_intrin.width || other_y1 >= other_intrin.height)
                        continue;

                    // Transfer between the depth pixels and the pixels inside the rectangle on the other image
                    for (int y = other_y0; y <= other_y1; ++y)
                    {
                        for (int x = other_x0; x <= other_x1; ++x)
                        {
                            transfer_pixel(depth_pixel_index, y * other_intrin.width + x);
                        }
                    }
                }
            }
        }
    }

    // Inverse Mapping with nearest interpolation
    template<class GET_DEPTH, class TRANSFER_PIXEL>
    void align_images_nearest(
        const rs2_intrinsics& depth_intrin,
        const rs2_extrinsics& other_to_depth,
        const rs2_intrinsics& other_intrin,
        GET_DEPTH get_depth,
        TRANSFER_PIXEL transfer_pixel)
    {
        // Iterate over the pixels of the aligned depth image
#pragma omp parallel for schedule(dynamic)
        for (int other_y = 0; other_y < other_intrin.height; ++other_y)
        {
            int other_pixel_index = other_y * other_intrin.width;
            for (int other_x = 0; other_x < other_intrin.width; ++other_x, ++other_pixel_index)
            {
                float other_pixel[2] = { other_x, other_y }, other_point[3], depth_pixel[2], depth_point[3];
                rs2_deproject_pixel_to_point(
                    other_point,
                    &other_intrin,
                    other_pixel,
                    1.0 // temp depth = 1.0
                );
                rs2_transform_point_to_point(depth_point, &other_to_depth, other_point);
                rs2_project_point_to_pixel(depth_pixel, &depth_intrin, depth_point);

                const int depth_x1 = static_cast<int>(depth_pixel[0] + 0.5f);
                const int depth_y1 = static_cast<int>(depth_pixel[1] + 0.5f);
                if( depth_x1 >= 0 && depth_x1 < depth_intrin.width && depth_y1 >= 0 && depth_y1 < depth_intrin.height )
                {
                    transfer_pixel(
                        depth_y1*depth_intrin.width + depth_x1,
                        other_pixel_index
                    );
                }
            }
        }
    }

    // Inverse Mapping with bilinear interpolation
    template<class GET_DEPTH>
    void align_images_bilinear(
        const rs2_intrinsics& depth_intrin,
        const rs2_extrinsics& other_to_depth,
        const rs2_intrinsics& other_intrin,
        GET_DEPTH get_depth_pixel,
        float z_scale,
        uint16_t * out_z)
    {
        // Iterate over the pixels of the aligned depth image
        #pragma omp parallel for schedule(dynamic)
        for (int other_y = 0; other_y < other_intrin.height; ++other_y)
        {
            int other_pixel_index = other_y * other_intrin.width;
            for (int other_x = 0; other_x < other_intrin.width; ++other_x, ++other_pixel_index)
            {
                float other_pixel[2] = { other_x, other_y }, other_point[3], depth_pixel[2], depth_point[3];
                rs2_deproject_pixel_to_point(
                    other_point,
                    &other_intrin,
                    other_pixel,
                    1.0 // temp depth = 1.0
                );
                rs2_transform_point_to_point(depth_point, &other_to_depth, other_point);
                rs2_project_point_to_pixel(depth_pixel, &depth_intrin, depth_point);

                double alpha, beta;
                int xa, xb, xc, xd, ya, yb, yc, yd;
                if( depth_pixel[0] >= 0.0 && depth_pixel[0] < (float)depth_intrin.width && 
                    depth_pixel[1] >= 0.0 && depth_pixel[1] < (float)depth_intrin.height)
                {
                    xa = floor(depth_pixel[0]); ya = floor(depth_pixel[1]);
                    xb = ceil(depth_pixel[0]);  yb = floor(depth_pixel[1]);
                    xc = floor(depth_pixel[0]); yc = ceil(depth_pixel[1]);
                    xd = ceil(depth_pixel[0]);  yd = ceil(depth_pixel[1]);
                    alpha = depth_pixel[0] - xa;
                    beta = depth_pixel[1] - ya;

                    out_z[other_pixel_index] = 
                        (
                            (1-alpha)*(1-beta)*get_depth_pixel(ya*depth_intrin.width+xa) +
                            (  alpha)*(1-beta)*get_depth_pixel(yb*depth_intrin.width+xb) +
                            (1-alpha)*(  beta)*get_depth_pixel(yc*depth_intrin.width+xc) +
                            (  alpha)*(  beta)*get_depth_pixel(yd*depth_intrin.width+xd) 
                        );
                }
            }
        }
    }

    // temp code, will be refactored
    std::vector<std::vector<float>> g_kernel;

    std::vector<std::vector<float>> makeIDWKernel(const int& size)
    {
        std::vector<std::vector<float>> ret(size, std::vector<float>(size, 0.0));
        if( size % 2 == 0 )
        {
            printf("[E] kernel size(%d) not even!\n", size);
            return ret;
        }

        for(int y=0 ; y<size ; ++y)
        {
            for(int x=0 ; x<size ; ++x)
            {
                ret[y][x] = 1.0 / sqrt(
                    (y-(size/2))*(y-(size/2))+
                    (x-(size/2))*(x-(size/2))
                );
                // printf("[%.2f]", ret[y][x]);
            }
            // printf("\n");
        }
        return ret;
    }

    uint16_t get_IDW_value(const int& x, const int& y, std::vector<std::vector<float>>& kernel, const uint16_t* out_z, const int& src_width, const int& src_height)
    {
        // printf("get_IDW_value [%d][%d]\n", x, y);
        float weighted_sum = 0;
        float total_dist = 0;
#pragma omp parallel for schedule(dynamic)
        for(size_t ky = 0 ; ky < kernel.size() ; ++ky)
        {
            int ky_offset = ky - kernel.size()/2;
            for(size_t kx = 0 ; kx < kernel.front().size() ; ++kx)
            {
                int kx_offset = kx - kernel.front().size()/2;
                if( y + ky_offset < 0 || x + kx_offset < 0 || y + ky_offset >= src_height || x + kx_offset >= src_width )
                {
                    continue;
                }
                uint16_t ref_val = out_z[(y + ky_offset)*src_width+(x + kx_offset)];
                if( ref_val )
                {
                    weighted_sum += kernel[ky][kx]*(float)ref_val;
                    total_dist += kernel[ky][kx];
                }
            }
        }
        return ( total_dist == 0 ) ? 0 : weighted_sum/total_dist;
    }

    template<class GET_DEPTH, class TRANSFER_PIXEL>
    void align_images_forward_IDW(
        const rs2_intrinsics& depth_intrin,
        const rs2_extrinsics& depth_to_other,
        const rs2_intrinsics& other_intrin,
        GET_DEPTH get_depth,
        TRANSFER_PIXEL transfer_pixel,
        float z_scale,
        uint16_t* out_z,
        uint16_t* p_working_buf)
    {
        // Iterate over the pixels of the depth image
#pragma omp parallel for schedule(dynamic)
        // phase 1 : forward mapping
#if 1
        for (int depth_y = 0; depth_y < depth_intrin.height; ++depth_y)
        {
            int depth_pixel_index = depth_y * depth_intrin.width;
            for (int depth_x = 0; depth_x < depth_intrin.width; ++depth_x, ++depth_pixel_index)
            {
                // Skip over depth pixels with the value of zero, we have no depth data so we will not write anything into our aligned images
                if (float depth = get_depth(depth_pixel_index))
                {
                    // Map the top-left corner of the depth pixel onto the other image
                    float depth_pixel[2] = { depth_x , depth_y }, depth_point[3], other_point[3], other_pixel[2];
                    rs2_deproject_pixel_to_point(depth_point, &depth_intrin, depth_pixel, depth);
                    rs2_transform_point_to_point(other_point, &depth_to_other, depth_point);
                    rs2_project_point_to_pixel(other_pixel, &other_intrin, other_point);
                    const int other_x0 = static_cast<int>(other_pixel[0] + 0.5f); //rounding
                    const int other_y0 = static_cast<int>(other_pixel[1] + 0.5f);

                    if (other_x0 < 0 || other_y0 < 0 || other_x0 >= other_intrin.width || other_y0 >= other_intrin.height)
                    {
                        continue;
                    }
                    transfer_pixel(depth_pixel_index, other_y0 * other_intrin.width + other_x0);
                }
            }
        }   
#else
        for (int other_y = 0; other_y < other_intrin.height; ++other_y)
        {
            int other_pixel_index = other_y * other_intrin.width;
            for (int other_x = 0; other_x < other_intrin.width; ++other_x, ++other_pixel_index)
            {
                if( other_y % 10 != 0 )
                {
                    out_z[other_pixel_index] = 0.0;
                }
                else
                {
                    out_z[other_pixel_index] = 0.5/z_scale;
                }
            }
        }
#endif     
        // phase 2 : IDW interpolation
        for (int other_y = 0; other_y < other_intrin.height; ++other_y)
        {
            int other_pixel_index = other_y * other_intrin.width;
            for (int other_x = 0; other_x < other_intrin.width; ++other_x, ++other_pixel_index)
            {
                if( !out_z[other_pixel_index] )
                {
                    // produce interpolation result and store in p_working_buf
                    p_working_buf[other_pixel_index] = get_IDW_value(
                        other_x, other_y,
                        g_kernel,
                        out_z, other_intrin.width, other_intrin.height                       
                    );
                }
            }
        }

        // for (int other_y = 0; other_y < other_intrin.height; ++other_y)
        // {
        //     int other_pixel_index = other_y * other_intrin.width;
        //     for (int other_x = 0; other_x < other_intrin.width; ++other_x, ++other_pixel_index)
        //     {
        //         if( !out_z[other_pixel_index] )
        //         {
        //             out_z[other_pixel_index] = p_working_buf[other_pixel_index];
        //         }
        //     }
        // }
    }

    void align_z_to_other(
        byte* z_aligned_to_other,
        const uint16_t* z_pixels,
        float z_scale,
        const rs2_intrinsics& z_intrin,
        const rs2_extrinsics& trans,
        const rs2_intrinsics& other_intrin,
        uint16_t* p_working_buf
    ){
        auto out_z = (uint16_t *)(z_aligned_to_other);

        auto get_depth  = [z_pixels, z_scale](int z_pixel_index)
        {
            return z_scale * z_pixels[z_pixel_index];
        };

        auto get_depth_pixel  = [z_pixels, z_scale](int z_pixel_index)
        {
            return z_pixels[z_pixel_index];
        };

        auto transfer_pixel = [out_z, z_pixels](int z_pixel_index, int other_pixel_index)
        {
            out_z[other_pixel_index] = out_z[other_pixel_index] ?
                // std::min((int)out_z[other_pixel_index], (int)z_pixels[z_pixel_index]) :
                out_z[other_pixel_index]:
                z_pixels[z_pixel_index];
        };

        switch(ALIGN_METHOD){
            case 0:
                align_images_nearest(z_intrin, trans, other_intrin, get_depth, transfer_pixel);
                break;
            case 1:
                align_images_bilinear(z_intrin, trans, other_intrin, get_depth_pixel, z_scale, out_z);
                break;
            case 2:
                std::memset(p_working_buf, 0, other_intrin.width*other_intrin.height*sizeof(uint16_t));
                align_images_forward_IDW(z_intrin, trans, other_intrin, get_depth, transfer_pixel, z_scale, out_z, p_working_buf);
                break;
            default:
                // RealSense stock alg
                align_images(z_intrin, trans, other_intrin, get_depth, transfer_pixel);
        }    
    }

    template<int N> struct bytes { char b[N]; };

    template<int N, class GET_DEPTH>
    void align_other_to_depth_bytes(byte* other_aligned_to_depth, GET_DEPTH get_depth, const rs2_intrinsics& depth_intrin, const rs2_extrinsics& depth_to_other, const rs2_intrinsics& other_intrin, const byte* other_pixels)
    {
        auto in_other = (const bytes<N> *)(other_pixels);
        auto out_other = (bytes<N> *)(other_aligned_to_depth);
        align_images(depth_intrin, depth_to_other, other_intrin, get_depth,
            [out_other, in_other](int depth_pixel_index, int other_pixel_index) { out_other[depth_pixel_index] = in_other[other_pixel_index]; });
    }

    template<class GET_DEPTH>
    void align_other_to_depth(byte* other_aligned_to_depth, GET_DEPTH get_depth, const rs2_intrinsics& depth_intrin, const rs2_extrinsics & depth_to_other, const rs2_intrinsics& other_intrin, const byte* other_pixels, rs2_format other_format)
    {
        switch (other_format)
        {
        case RS2_FORMAT_Y8:
            align_other_to_depth_bytes<1>(other_aligned_to_depth, get_depth, depth_intrin, depth_to_other, other_intrin, other_pixels);
            break;
        case RS2_FORMAT_Y16:
        case RS2_FORMAT_Z16:
            align_other_to_depth_bytes<2>(other_aligned_to_depth, get_depth, depth_intrin, depth_to_other, other_intrin, other_pixels);
            break;
        case RS2_FORMAT_RGB8:
        case RS2_FORMAT_BGR8:
            align_other_to_depth_bytes<3>(other_aligned_to_depth, get_depth, depth_intrin, depth_to_other, other_intrin, other_pixels);
            break;
        case RS2_FORMAT_RGBA8:
        case RS2_FORMAT_BGRA8:
            align_other_to_depth_bytes<4>(other_aligned_to_depth, get_depth, depth_intrin, depth_to_other, other_intrin, other_pixels);
            break;
        default:
            assert(false); // NOTE: rs2_align_other_to_depth_bytes<2>(...) is not appropriate for RS2_FORMAT_YUYV/RS2_FORMAT_RAW10 images, no logic prevents U/V channels from being written to one another
        }
    }

    void align_other_to_z(byte* other_aligned_to_z, const uint16_t* z_pixels, float z_scale, const rs2_intrinsics& z_intrin, const rs2_extrinsics& z_to_other, const rs2_intrinsics& other_intrin, const byte* other_pixels, rs2_format other_format)
    {
        align_other_to_depth(other_aligned_to_z, [z_pixels, z_scale](int z_pixel_index) { return z_scale * z_pixels[z_pixel_index]; }, z_intrin, z_to_other, other_intrin, other_pixels, other_format);
    }

    int align::get_unique_id(const std::shared_ptr<stream_profile_interface>& original_profile,
        const std::shared_ptr<stream_profile_interface>& to_profile,
        const std::shared_ptr<stream_profile_interface>& aligned_profile)
    {
        //align_stream_unique_ids holds a cache of mapping between the 2 streams that created the new aligned stream
        // to it stream id.
        //When an aligned frame is created from other streams (but with the same instance of this class)
        // the from_to pair will be different so a new id will be added to the cache.
        //This allows the user to pass different streams to this class and for every pair of from_to
        //the user will always get the same stream id for the aligned stream.
        auto from_to = std::make_pair(original_profile->get_unique_id(), to_profile->get_unique_id());
        auto it = align_stream_unique_ids.find(from_to);
        if (it != align_stream_unique_ids.end())
        {
            return it->second;
        }
        else
        {
            int new_id = aligned_profile->get_unique_id();
            align_stream_unique_ids[from_to] = new_id;
            return new_id;
        }
    }
    std::shared_ptr<stream_profile_interface> align::create_aligned_profile(
        const std::shared_ptr<stream_profile_interface>& original_profile,
        const std::shared_ptr<stream_profile_interface>& to_profile)
    {
        auto aligned_profile = original_profile->clone();
        int aligned_unique_id = get_unique_id(original_profile, to_profile, aligned_profile);
        aligned_profile->set_unique_id(aligned_unique_id);
        environment::get_instance().get_extrinsics_graph().register_same_extrinsics(*aligned_profile, *original_profile);
        aligned_profile->set_stream_index(original_profile->get_stream_index());
        aligned_profile->set_stream_type(original_profile->get_stream_type());
        aligned_profile->set_format(original_profile->get_format());
        aligned_profile->set_framerate(original_profile->get_framerate());
        if (auto original_video_profile = As<video_stream_profile_interface>(original_profile))
        {
            if (auto to_video_profile = As<video_stream_profile_interface>(to_profile))
            {
                if (auto aligned_video_profile = As<video_stream_profile_interface>(aligned_profile))
                {
                    aligned_video_profile->set_dims(to_video_profile->get_width(), to_video_profile->get_height());
                    auto aligned_intrinsics = original_video_profile->get_intrinsics();
                    aligned_intrinsics.width = to_video_profile->get_width();
                    aligned_intrinsics.height = to_video_profile->get_height();
                    aligned_video_profile->set_intrinsics([aligned_intrinsics]() { return aligned_intrinsics; });
                }
            }
        }
        return aligned_profile;
    }
    void align::on_frame(frame_holder frameset, librealsense::synthetic_source_interface* source)
    {
        auto composite = As<composite_frame>(frameset.frame);
        if (composite == nullptr)
        {
            LOG_WARNING("Trying to align a non composite frame");
            return;
        }

        if (composite->get_embedded_frames_count() < 2)
        {
            LOG_WARNING("Trying to align a single frame");
            return;
        }

        librealsense::video_frame* depth_frame = nullptr;
        std::vector<librealsense::video_frame*> other_frames;
        //Find the depth frame
        for (int i = 0; i < composite->get_embedded_frames_count(); i++)
        {
            frame_interface* f = composite->get_frame(i);
            if (f->get_stream()->get_stream_type() == RS2_STREAM_DEPTH)
            {
                assert(depth_frame == nullptr); // Trying to align multiple depth frames is not supported, in release we take the last one
                depth_frame = As<librealsense::video_frame>(f);
                if (depth_frame == nullptr)
                {
                    LOG_ERROR("Given depth frame is not a librealsense::video_frame");
                    return;
                }
            }
            else
            {
                auto other_video_frame = As<librealsense::video_frame>(f);
                auto other_stream_profile = f->get_stream();
                assert(other_stream_profile != nullptr);

                if (other_video_frame == nullptr)
                {
                    LOG_ERROR("Given frame of type " << other_stream_profile->get_stream_type() << ", is not a librealsense::video_frame, ignoring it");
                    return;
                }

                if (_to_stream_type == RS2_STREAM_DEPTH)
                {
                    //In case of alignment to depth, we will align any image given in the frameset to the depth one
                    other_frames.push_back(other_video_frame);
                }
                else
                {
                    //In case of alignment from depth to other, we only want the other frame with the stream type that was requested to align to
                    if (other_stream_profile->get_stream_type() == _to_stream_type)
                    {
                        assert(other_frames.size() == 0); // Trying to align depth to multiple frames is not supported, in release we take the last one
                        other_frames.push_back(other_video_frame);
                    }
                }
            }
        }

        if (depth_frame == nullptr)
        {
            LOG_WARNING("No depth frame provided to align");
            return;
        }

        if (other_frames.empty())
        {
            LOG_WARNING("Only depth frame provided to align");
            return;
        }

        auto depth_profile = As<video_stream_profile_interface>(depth_frame->get_stream());
        if (depth_profile == nullptr)
        {
            LOG_ERROR("Depth profile is not a video stream profile");
            return;
        }
        rs2_intrinsics depth_intrinsics = depth_profile->get_intrinsics();
        std::vector<frame_holder> output_frames;

        if (_to_stream_type == RS2_STREAM_DEPTH)
        {
            //Storing the original depth frame for output frameset
            depth_frame->acquire();
            output_frames.push_back(depth_frame);
        }

        for (librealsense::video_frame* other_frame : other_frames)
        {
            auto other_profile = As<video_stream_profile_interface>(other_frame->get_stream());
            if (other_profile == nullptr)
            {
                LOG_WARNING("Other frame with type " << other_frame->get_stream()->get_stream_type() << ", is not a video stream profile. Ignoring it");
                continue;
            }

            rs2_intrinsics other_intrinsics = other_profile->get_intrinsics();
            rs2_extrinsics depth_to_other_extrinsics{};
            rs2_extrinsics other_to_depth_extrinsics{};
            if (!environment::get_instance().get_extrinsics_graph().try_fetch_extrinsics(*depth_profile, *other_profile, &depth_to_other_extrinsics))
            {
                LOG_WARNING("Failed to get extrinsics from depth to " << other_profile->get_stream_type() << ", ignoring it");
                continue;
            }
            if (!environment::get_instance().get_extrinsics_graph().try_fetch_extrinsics(*other_profile, *depth_profile, &other_to_depth_extrinsics))
            {
                LOG_WARNING("Failed to get extrinsics from other to " << depth_profile->get_stream_type() << ", ignoring it");
                continue;
            }

            auto sensor = depth_frame->get_sensor();
            if (sensor == nullptr)
            {
                LOG_ERROR("Failed to get sensor from depth frame");
                return;
            }

            if (sensor->supports_option(RS2_OPTION_DEPTH_UNITS) == false)
            {
                LOG_ERROR("Sensor of depth frame does not provide depth units");
                return;
            }

            float depth_scale = sensor->get_option(RS2_OPTION_DEPTH_UNITS).query();

            frame_holder aligned_frame{ nullptr };
            if (_to_stream_type == RS2_STREAM_DEPTH)
            {
                //Align a stream to depth
                auto aligned_bytes_per_pixel = other_frame->get_bpp() / 8;
                auto aligned_profile = create_aligned_profile(other_profile, depth_profile);
                aligned_frame = source->allocate_video_frame(
                    aligned_profile,
                    other_frame,
                    aligned_bytes_per_pixel,
                    depth_frame->get_width(),
                    depth_frame->get_height(),
                    depth_frame->get_width() * aligned_bytes_per_pixel,
                    RS2_EXTENSION_VIDEO_FRAME);

                if (aligned_frame == nullptr)
                {
                    LOG_ERROR("Failed to allocate frame for aligned output");
                    return;
                }

                byte* other_aligned_to_depth = const_cast<byte*>(aligned_frame.frame->get_frame_data());
                memset(other_aligned_to_depth, 0, depth_intrinsics.height * depth_intrinsics.width * aligned_bytes_per_pixel);
                align_other_to_z(other_aligned_to_depth,
                    reinterpret_cast<const uint16_t*>(depth_frame->get_frame_data()),
                    depth_scale, depth_intrinsics,
                    depth_to_other_extrinsics,
                    other_intrinsics,
                    other_frame->get_frame_data(),
                    other_profile->get_format());
            }
            else
            {
                //Align depth to some stream
                auto aligned_bytes_per_pixel = depth_frame->get_bpp() / 8;
                auto aligned_profile = create_aligned_profile(depth_profile, other_profile);
                aligned_frame = source->allocate_video_frame(
                    aligned_profile,
                    depth_frame,
                    aligned_bytes_per_pixel,
                    other_intrinsics.width,
                    other_intrinsics.height,
                    other_intrinsics.width * aligned_bytes_per_pixel,
                    RS2_EXTENSION_DEPTH_FRAME);

                if (aligned_frame == nullptr)
                {
                    LOG_ERROR("Failed to allocate frame for aligned output");
                    return;
                }
                byte* z_aligned_to_other = const_cast<byte*>(aligned_frame.frame->get_frame_data());
                memset(z_aligned_to_other, 0, other_intrinsics.height * other_intrinsics.width * aligned_bytes_per_pixel);
                align_z_to_other(z_aligned_to_other,
                    reinterpret_cast<const uint16_t*>(depth_frame->get_frame_data()),
                    depth_scale,
                    depth_intrinsics,
                    depth_to_other_extrinsics,
                    other_intrinsics,
                    mp_working_buf);

                //Storing the original other frame for output frameset
                assert(output_frames.size() == 0); //When aligning depth to other, only 2 frames are expected in the output.
                other_frame->acquire();
                output_frames.push_back(other_frame);
            }
            output_frames.push_back(std::move(aligned_frame));
        }
        auto new_composite = source->allocate_composite_frame(std::move(output_frames));
        source->frame_ready(std::move(new_composite));
    }

    align::align(rs2_stream to_stream) : _to_stream_type(to_stream)
    {
        printf("Align method(%d)\n", ALIGN_METHOD);
        auto cb = [this](frame_holder frameset, librealsense::synthetic_source_interface* source) { on_frame(std::move(frameset), source); };
        auto callback = new internal_frame_processor_callback<decltype(cb)>(cb);
        processing_block::set_processing_callback(std::shared_ptr<rs2_frame_processor_callback>(callback));

        // tmp hard coded
        // uint16_t* mp_working_buf = new uint16_t[other_intrin.width*other_intrin.height];
        mp_working_buf = new uint16_t[1920*1080];
        // where can I release this buffer?

        int kernel_size = 5;
        g_kernel = makeIDWKernel(kernel_size);
        printf("kernel size [%d][%d]\n", g_kernel.size(), g_kernel.front().size());
    }
}
