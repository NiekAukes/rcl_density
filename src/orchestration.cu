// Auto-generated CUDA orchestrator — do not edit
#pragma once

#include "density_function.cu"
#include "helpers.cu"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cstdio>

static const int GRID_X = 16;
static const int GRID_Y = 2096;
static const int GRID_Z = 16;
static const int TOTAL_ELEMENTS = 536576; // 16 * 2096 * 16
static const size_t BUFFER_SIZE = (size_t)TOTAL_ELEMENTS * sizeof(double);

// ============================================================================
// CUDA PIPELINE: final_density
// ============================================================================
class CudaPipeline_final_density {
private:
    int3   grid_size;
    int    total_elements;
    size_t buffer_size;
    cudaStream_t stream; // dedicated stream so instances run concurrently

    // Output buffers (one per kernel)
    double* d_minecraft_jagged_27_d5x132x5os16391250x0x16391250ps65565000x0x65565000_output;
    double* d_minecraft_gravel_34_d5x1x5os3278x0x3278ps13113x0x13113_output;
    double* d_minecraft_cave_layer_16_d5x1x5os13113x0x13113ps52452x0x52452_output;
    double* d_minecraft_cave_layer_52_d5x132x5os32782x32782x32782ps131130x524520x131130_output;
    double* d_minecraft_cave_layer_30_d5x1x5os6556x0x6556ps26226x0x26226_output;
    double* d_minecraft_jagged_7_d5x1x5os39339000x0x39339000ps157356000x0x157356000_output;
    double* d_minecraft_jagged_27_d5x1x5os16391250x0x16391250ps65565000x0x65565000_output;
    double* d_density_function_ShiftedNoise_3_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_cave_layer_37_d5x1x5os393x0x393ps1573x0x1573_output;
    double* d_density_function_ShiftedNoise_20_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_gravel_39_d5x1x5os524x0x524ps2098x0x2098_output;
    double* d_density_function_YClampedGradient_53_d5x132x5os65565x65565x65565ps262260x1049040x262260_output;
    double* d_minecraft_jagged_15_d5x1x5os32848065x0x32848065ps131392260x0x131392260_output;
    double* d_minecraft_cave_layer_46_d5x132x5os262260x262260x262260ps1049040x4196160x1049040_output;
    double* d_minecraft_gravel_9_d5x1x5os7212x0x7212ps28848x0x28848_output;
    double* d_minecraft_gravel_31_d5x1x5os6556x0x6556ps26226x0x26226_output;
    double* d_density_function_ShiftedNoise_11_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_jagged_49_d5x132x5os65565000x0x65565000ps262260000x0x262260000_output;
    double* d_minecraft_jagged_32_d5x132x5os32782500x0x32782500ps131130000x0x131130000_output;
    double* d_minecraft_jagged_24_d5x1x5os19669500x0x19669500ps78678000x0x78678000_output;
    double* d_minecraft_cave_layer_51_d5x132x5os163912x65565x163912ps655650x1049040x655650_output;
    double* d_minecraft_cave_layer_50_d5x132x5os327825x131130x327825ps1311300x2098080x1311300_output;
    double* d_minecraft_gravel_28_d5x1x5os13113x0x13113ps52452x0x52452_output;
    double* d_minecraft_cave_layer_8_d5x1x5os19669x0x19669ps78678x0x78678_output;
    double* d_minecraft_cave_layer_42_d5x132x5os131130x0x131130ps524520x0x524520_output;
    double* d_minecraft_gravel_1_d5x1x5os7867x0x7867ps31471x0x31471_output;
    double* d_density_function_ShiftedNoise_13_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_cave_layer_43_d5x132x5os65565x65565x65565ps262260x1049040x262260_output;
    double* d_minecraft_gravel_47_d5x132x5os1966x0x1966ps7867x0x7867_output;
    double* d_minecraft_jagged_35_d5x1x5os11473875x0x11473875ps45895500x0x45895500_output;
    double* d_density_function_ShiftedNoise_22_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_gravel_17_d5x1x5os16391x0x16391ps65565x0x65565_output;
    double* d_minecraft_cave_layer_25_d5x1x5os9834x0x9834ps39339x0x39339_output;
    double* d_density_function_ShiftedNoise_5_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_cave_layer_0_d5x1x5os16391x0x16391ps65565x0x65565_output;
    double* d_minecraft_cave_layer_45_d5x132x5os393390x393390x393390ps1573560x6294240x1573560_output;
    double* d_minecraft_jagged_32_d5x1x5os32782500x0x32782500ps131130000x0x131130000_output;
    double* d_minecraft_cave_layer_44_d5x132x5os524520x524520x524520ps2098080x8392320x2098080_output;
    double* d_minecraft_cave_layer_48_d5x132x5os655650x262260x655650ps2622600x4196160x2622600_output;
    double* d_minecraft_realism_mountains_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_density_function_Multiply_10_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_density_function_Multiply_2_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_density_function_Add_33_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_caves_barrier_1_d5x132x5os65565x65565x65565ps262260x1049040x262260_output;
    double* d_minecraft_caves_old_small_d5x132x5os65565x65565x65565ps262260x1049040x262260_output;
    double* d_density_function_Add_29_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_realism_mountains_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_realism_extreme_mountains_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_caves_barrier_2_d5x132x5os65565x65565x65565ps262260x1049040x262260_output;
    double* d_density_function_Noise_41_d5x132x5os65565x65565x65565ps262260x1049040x262260_output;
    double* d_minecraft_caves_old_medium_d5x132x5os65565x65565x65565ps262260x1049040x262260_output;
    double* d_density_function_Add_38_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_realism_extreme_mountains_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_determiner_overworld_1_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_realism_hills_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_realism_extreme_hills_factor_divergence_squared_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_realism_hills_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_determiner_cave_1_d5x132x5os65565x65565x65565ps262260x1049040x262260_output;
    double* d_minecraft_realism_extreme_hills_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_realism_extreme_hills_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_caves_old_large_d5x132x5os65565x65565x65565ps262260x1049040x262260_output;
    double* d_minecraft_realism_extreme_mountains_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_density_function_Abs_36_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_caves_new_small_d5x132x5os65565x65565x65565ps262260x1049040x262260_output;
    double* d_density_function_Multiply_19_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_caves_new_medium_d5x132x5os65565x65565x65565ps262260x1049040x262260_output;
    double* d_minecraft_realism_mountains_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_caves_new_large_d5x132x5os65565x65565x65565ps262260x1049040x262260_output;
    double* d_density_function_Multiply_18_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_realism_extreme_hills_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_realism_hills_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_new_surface_height_unprocessed_unrivered_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_new_surface_height_unprocessed_rivered_d5x1x5os65565x0x65565ps262260x0x262260_output;
    double* d_minecraft_caves_underlands_combined_d5x132x5os65565x65565x65565ps262260x1049040x262260_output;
    double* d_minecraft_new_surface_combined_processed_d5x132x5os65565x65565x65565ps262260x1049040x262260_output;
    double* d_minecraft_new_combination_unfixed_d5x132x5os65565x65565x65565ps262260x1049040x262260_output;
    double* d_minecraft_new_combination_fixed_d5x132x5os65565x65565x65565ps262260x1049040x262260_output;
    double* d_final_density_d16x2096x16os65565x65565x65565ps65565x65565x65565_output;

    // Permutation tables (deduplicated across all kernels)
    int8_t* d_perm_table_minecraft_cave_layer_0_octave__8;
    int8_t* d_perm_table_minecraft_cave_layer_1_octave__8;
    int8_t* d_perm_table_minecraft_gravel_0_octave__5;
    int8_t* d_perm_table_minecraft_gravel_1_octave__5;
    int8_t* d_perm_table_minecraft_gravel_0_octave__6;
    int8_t* d_perm_table_minecraft_gravel_1_octave__6;
    int8_t* d_perm_table_minecraft_gravel_0_octave__7;
    int8_t* d_perm_table_minecraft_gravel_1_octave__7;
    int8_t* d_perm_table_minecraft_gravel_0_octave__8;
    int8_t* d_perm_table_minecraft_gravel_1_octave__8;
    int8_t* d_perm_table_minecraft_jagged_0_octave__1;
    int8_t* d_perm_table_minecraft_jagged_1_octave__1;
    int8_t* d_perm_table_minecraft_jagged_0_octave__10;
    int8_t* d_perm_table_minecraft_jagged_1_octave__10;
    int8_t* d_perm_table_minecraft_jagged_0_octave__11;
    int8_t* d_perm_table_minecraft_jagged_1_octave__11;
    int8_t* d_perm_table_minecraft_jagged_0_octave__12;
    int8_t* d_perm_table_minecraft_jagged_1_octave__12;
    int8_t* d_perm_table_minecraft_jagged_0_octave__13;
    int8_t* d_perm_table_minecraft_jagged_1_octave__13;
    int8_t* d_perm_table_minecraft_jagged_0_octave__14;
    int8_t* d_perm_table_minecraft_jagged_1_octave__14;
    int8_t* d_perm_table_minecraft_jagged_0_octave__15;
    int8_t* d_perm_table_minecraft_jagged_1_octave__15;
    int8_t* d_perm_table_minecraft_jagged_0_octave__16;
    int8_t* d_perm_table_minecraft_jagged_1_octave__16;
    int8_t* d_perm_table_minecraft_jagged_0_octave__2;
    int8_t* d_perm_table_minecraft_jagged_1_octave__2;
    int8_t* d_perm_table_minecraft_jagged_0_octave__3;
    int8_t* d_perm_table_minecraft_jagged_1_octave__3;
    int8_t* d_perm_table_minecraft_jagged_0_octave__4;
    int8_t* d_perm_table_minecraft_jagged_1_octave__4;
    int8_t* d_perm_table_minecraft_jagged_0_octave__5;
    int8_t* d_perm_table_minecraft_jagged_1_octave__5;
    int8_t* d_perm_table_minecraft_jagged_0_octave__6;
    int8_t* d_perm_table_minecraft_jagged_1_octave__6;
    int8_t* d_perm_table_minecraft_jagged_0_octave__7;
    int8_t* d_perm_table_minecraft_jagged_1_octave__7;
    int8_t* d_perm_table_minecraft_jagged_0_octave__8;
    int8_t* d_perm_table_minecraft_jagged_1_octave__8;
    int8_t* d_perm_table_minecraft_jagged_0_octave__9;
    int8_t* d_perm_table_minecraft_jagged_1_octave__9;

public:
    CudaPipeline_final_density(int64_t world_seed) {
        grid_size      = make_int3(GRID_X, GRID_Y, GRID_Z);
        total_elements = TOTAL_ELEMENTS;
        buffer_size    = BUFFER_SIZE;
        cudaStreamCreate(&stream);

        // Allocate output buffers
        cudaMalloc(&d_minecraft_jagged_27_d5x132x5os16391250x0x16391250ps65565000x0x65565000_output, buffer_size);
        cudaMalloc(&d_minecraft_gravel_34_d5x1x5os3278x0x3278ps13113x0x13113_output, buffer_size);
        cudaMalloc(&d_minecraft_cave_layer_16_d5x1x5os13113x0x13113ps52452x0x52452_output, buffer_size);
        cudaMalloc(&d_minecraft_cave_layer_52_d5x132x5os32782x32782x32782ps131130x524520x131130_output, buffer_size);
        cudaMalloc(&d_minecraft_cave_layer_30_d5x1x5os6556x0x6556ps26226x0x26226_output, buffer_size);
        cudaMalloc(&d_minecraft_jagged_7_d5x1x5os39339000x0x39339000ps157356000x0x157356000_output, buffer_size);
        cudaMalloc(&d_minecraft_jagged_27_d5x1x5os16391250x0x16391250ps65565000x0x65565000_output, buffer_size);
        cudaMalloc(&d_density_function_ShiftedNoise_3_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_cave_layer_37_d5x1x5os393x0x393ps1573x0x1573_output, buffer_size);
        cudaMalloc(&d_density_function_ShiftedNoise_20_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_gravel_39_d5x1x5os524x0x524ps2098x0x2098_output, buffer_size);
        cudaMalloc(&d_density_function_YClampedGradient_53_d5x132x5os65565x65565x65565ps262260x1049040x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_jagged_15_d5x1x5os32848065x0x32848065ps131392260x0x131392260_output, buffer_size);
        cudaMalloc(&d_minecraft_cave_layer_46_d5x132x5os262260x262260x262260ps1049040x4196160x1049040_output, buffer_size);
        cudaMalloc(&d_minecraft_gravel_9_d5x1x5os7212x0x7212ps28848x0x28848_output, buffer_size);
        cudaMalloc(&d_minecraft_gravel_31_d5x1x5os6556x0x6556ps26226x0x26226_output, buffer_size);
        cudaMalloc(&d_density_function_ShiftedNoise_11_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_jagged_49_d5x132x5os65565000x0x65565000ps262260000x0x262260000_output, buffer_size);
        cudaMalloc(&d_minecraft_jagged_32_d5x132x5os32782500x0x32782500ps131130000x0x131130000_output, buffer_size);
        cudaMalloc(&d_minecraft_jagged_24_d5x1x5os19669500x0x19669500ps78678000x0x78678000_output, buffer_size);
        cudaMalloc(&d_minecraft_cave_layer_51_d5x132x5os163912x65565x163912ps655650x1049040x655650_output, buffer_size);
        cudaMalloc(&d_minecraft_cave_layer_50_d5x132x5os327825x131130x327825ps1311300x2098080x1311300_output, buffer_size);
        cudaMalloc(&d_minecraft_gravel_28_d5x1x5os13113x0x13113ps52452x0x52452_output, buffer_size);
        cudaMalloc(&d_minecraft_cave_layer_8_d5x1x5os19669x0x19669ps78678x0x78678_output, buffer_size);
        cudaMalloc(&d_minecraft_cave_layer_42_d5x132x5os131130x0x131130ps524520x0x524520_output, buffer_size);
        cudaMalloc(&d_minecraft_gravel_1_d5x1x5os7867x0x7867ps31471x0x31471_output, buffer_size);
        cudaMalloc(&d_density_function_ShiftedNoise_13_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_cave_layer_43_d5x132x5os65565x65565x65565ps262260x1049040x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_gravel_47_d5x132x5os1966x0x1966ps7867x0x7867_output, buffer_size);
        cudaMalloc(&d_minecraft_jagged_35_d5x1x5os11473875x0x11473875ps45895500x0x45895500_output, buffer_size);
        cudaMalloc(&d_density_function_ShiftedNoise_22_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_gravel_17_d5x1x5os16391x0x16391ps65565x0x65565_output, buffer_size);
        cudaMalloc(&d_minecraft_cave_layer_25_d5x1x5os9834x0x9834ps39339x0x39339_output, buffer_size);
        cudaMalloc(&d_density_function_ShiftedNoise_5_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_cave_layer_0_d5x1x5os16391x0x16391ps65565x0x65565_output, buffer_size);
        cudaMalloc(&d_minecraft_cave_layer_45_d5x132x5os393390x393390x393390ps1573560x6294240x1573560_output, buffer_size);
        cudaMalloc(&d_minecraft_jagged_32_d5x1x5os32782500x0x32782500ps131130000x0x131130000_output, buffer_size);
        cudaMalloc(&d_minecraft_cave_layer_44_d5x132x5os524520x524520x524520ps2098080x8392320x2098080_output, buffer_size);
        cudaMalloc(&d_minecraft_cave_layer_48_d5x132x5os655650x262260x655650ps2622600x4196160x2622600_output, buffer_size);
        cudaMalloc(&d_minecraft_realism_mountains_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_density_function_Multiply_10_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_density_function_Multiply_2_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_density_function_Add_33_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_caves_barrier_1_d5x132x5os65565x65565x65565ps262260x1049040x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_caves_old_small_d5x132x5os65565x65565x65565ps262260x1049040x262260_output, buffer_size);
        cudaMalloc(&d_density_function_Add_29_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_realism_mountains_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_realism_extreme_mountains_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_caves_barrier_2_d5x132x5os65565x65565x65565ps262260x1049040x262260_output, buffer_size);
        cudaMalloc(&d_density_function_Noise_41_d5x132x5os65565x65565x65565ps262260x1049040x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_caves_old_medium_d5x132x5os65565x65565x65565ps262260x1049040x262260_output, buffer_size);
        cudaMalloc(&d_density_function_Add_38_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_realism_extreme_mountains_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_determiner_overworld_1_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_realism_hills_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_realism_extreme_hills_factor_divergence_squared_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_realism_hills_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_determiner_cave_1_d5x132x5os65565x65565x65565ps262260x1049040x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_realism_extreme_hills_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_realism_extreme_hills_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_caves_old_large_d5x132x5os65565x65565x65565ps262260x1049040x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_realism_extreme_mountains_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_density_function_Abs_36_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_caves_new_small_d5x132x5os65565x65565x65565ps262260x1049040x262260_output, buffer_size);
        cudaMalloc(&d_density_function_Multiply_19_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_caves_new_medium_d5x132x5os65565x65565x65565ps262260x1049040x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_realism_mountains_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_caves_new_large_d5x132x5os65565x65565x65565ps262260x1049040x262260_output, buffer_size);
        cudaMalloc(&d_density_function_Multiply_18_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_realism_extreme_hills_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_realism_hills_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_new_surface_height_unprocessed_unrivered_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_new_surface_height_unprocessed_rivered_d5x1x5os65565x0x65565ps262260x0x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_caves_underlands_combined_d5x132x5os65565x65565x65565ps262260x1049040x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_new_surface_combined_processed_d5x132x5os65565x65565x65565ps262260x1049040x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_new_combination_unfixed_d5x132x5os65565x65565x65565ps262260x1049040x262260_output, buffer_size);
        cudaMalloc(&d_minecraft_new_combination_fixed_d5x132x5os65565x65565x65565ps262260x1049040x262260_output, buffer_size);
        cudaMalloc(&d_final_density_d16x2096x16os65565x65565x65565ps65565x65565x65565_output, buffer_size);

        // Allocate and initialize permutation tables from world seed
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0x4dfe67be2ef51a83), INT64_C(0x5a4ae8c4423e7206), // ident: "minecraft:cave_layer"
                INT64_C(0),
                INT64_C(0x0ef68ec68504005e), INT64_C(0x48b6bf93a2789640)  // subident: octave_-8
            );
            cudaMalloc(&d_perm_table_minecraft_cave_layer_0_octave__8, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_cave_layer_0_octave__8, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0x4dfe67be2ef51a83), INT64_C(0x5a4ae8c4423e7206), // ident: "minecraft:cave_layer"
                INT64_C(1),
                INT64_C(0x0ef68ec68504005e), INT64_C(0x48b6bf93a2789640)  // subident: octave_-8
            );
            cudaMalloc(&d_perm_table_minecraft_cave_layer_1_octave__8, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_cave_layer_1_octave__8, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0x1001fbe1934018b7), INT64_C(0x6ba60255f94d20e3), // ident: "minecraft:gravel"
                INT64_C(0),
                INT64_C(0x6d7b49e7e429850a), INT64_C(0x2e3063c622a24777)  // subident: octave_-5
            );
            cudaMalloc(&d_perm_table_minecraft_gravel_0_octave__5, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_gravel_0_octave__5, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0x1001fbe1934018b7), INT64_C(0x6ba60255f94d20e3), // ident: "minecraft:gravel"
                INT64_C(1),
                INT64_C(0x6d7b49e7e429850a), INT64_C(0x2e3063c622a24777)  // subident: octave_-5
            );
            cudaMalloc(&d_perm_table_minecraft_gravel_1_octave__5, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_gravel_1_octave__5, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0x1001fbe1934018b7), INT64_C(0x6ba60255f94d20e3), // ident: "minecraft:gravel"
                INT64_C(0),
                INT64_C(0xe51c98ce7d1de664), INT64_C(0x5f9478a733040c45)  // subident: octave_-6
            );
            cudaMalloc(&d_perm_table_minecraft_gravel_0_octave__6, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_gravel_0_octave__6, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0x1001fbe1934018b7), INT64_C(0x6ba60255f94d20e3), // ident: "minecraft:gravel"
                INT64_C(1),
                INT64_C(0xe51c98ce7d1de664), INT64_C(0x5f9478a733040c45)  // subident: octave_-6
            );
            cudaMalloc(&d_perm_table_minecraft_gravel_1_octave__6, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_gravel_1_octave__6, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0x1001fbe1934018b7), INT64_C(0x6ba60255f94d20e3), // ident: "minecraft:gravel"
                INT64_C(0),
                INT64_C(0xf11268128982754f), INT64_C(0x257a1d670430b0aa)  // subident: octave_-7
            );
            cudaMalloc(&d_perm_table_minecraft_gravel_0_octave__7, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_gravel_0_octave__7, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0x1001fbe1934018b7), INT64_C(0x6ba60255f94d20e3), // ident: "minecraft:gravel"
                INT64_C(1),
                INT64_C(0xf11268128982754f), INT64_C(0x257a1d670430b0aa)  // subident: octave_-7
            );
            cudaMalloc(&d_perm_table_minecraft_gravel_1_octave__7, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_gravel_1_octave__7, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0x1001fbe1934018b7), INT64_C(0x6ba60255f94d20e3), // ident: "minecraft:gravel"
                INT64_C(0),
                INT64_C(0x0ef68ec68504005e), INT64_C(0x48b6bf93a2789640)  // subident: octave_-8
            );
            cudaMalloc(&d_perm_table_minecraft_gravel_0_octave__8, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_gravel_0_octave__8, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0x1001fbe1934018b7), INT64_C(0x6ba60255f94d20e3), // ident: "minecraft:gravel"
                INT64_C(1),
                INT64_C(0x0ef68ec68504005e), INT64_C(0x48b6bf93a2789640)  // subident: octave_-8
            );
            cudaMalloc(&d_perm_table_minecraft_gravel_1_octave__8, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_gravel_1_octave__8, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(0),
                INT64_C(0xdffa22b534c5f608), INT64_C(0xb9b67517d3665ca9)  // subident: octave_-1
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_0_octave__1, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_0_octave__1, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(1),
                INT64_C(0xdffa22b534c5f608), INT64_C(0xb9b67517d3665ca9)  // subident: octave_-1
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_1_octave__1, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_1_octave__1, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(0),
                INT64_C(0x36d326eed40efeb2), INT64_C(0x5be9ce18223c636a)  // subident: octave_-10
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_0_octave__10, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_0_octave__10, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(1),
                INT64_C(0x36d326eed40efeb2), INT64_C(0x5be9ce18223c636a)  // subident: octave_-10
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_1_octave__10, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_1_octave__10, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(0),
                INT64_C(0x0fd787bfbc403ec3), INT64_C(0x74a4a31ca21b48b8)  // subident: octave_-11
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_0_octave__11, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_0_octave__11, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(1),
                INT64_C(0x0fd787bfbc403ec3), INT64_C(0x74a4a31ca21b48b8)  // subident: octave_-11
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_1_octave__11, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_1_octave__11, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(0),
                INT64_C(0xb198de63a8012672), INT64_C(0x7b84cad43ef7b5a8)  // subident: octave_-12
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_0_octave__12, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_0_octave__12, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(1),
                INT64_C(0xb198de63a8012672), INT64_C(0x7b84cad43ef7b5a8)  // subident: octave_-12
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_1_octave__12, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_1_octave__12, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(0),
                INT64_C(0xd1fc8a05be565eca), INT64_C(0xdc2a3915cbdda25b)  // subident: octave_-13
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_0_octave__13, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_0_octave__13, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(1),
                INT64_C(0xd1fc8a05be565eca), INT64_C(0xdc2a3915cbdda25b)  // subident: octave_-13
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_1_octave__13, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_1_octave__13, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(0),
                INT64_C(0xfc0027cef9683114), INT64_C(0xb758d3954dcbfdd3)  // subident: octave_-14
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_0_octave__14, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_0_octave__14, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(1),
                INT64_C(0xfc0027cef9683114), INT64_C(0xb758d3954dcbfdd3)  // subident: octave_-14
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_1_octave__14, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_1_octave__14, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(0),
                INT64_C(0x7eee475a921c6cf5), INT64_C(0xf2bd39426f8da413)  // subident: octave_-15
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_0_octave__15, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_0_octave__15, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(1),
                INT64_C(0x7eee475a921c6cf5), INT64_C(0xf2bd39426f8da413)  // subident: octave_-15
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_1_octave__15, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_1_octave__15, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(0),
                INT64_C(0xc613bf766619f992), INT64_C(0x954753f86691b86a)  // subident: octave_-16
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_0_octave__16, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_0_octave__16, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(1),
                INT64_C(0xc613bf766619f992), INT64_C(0x954753f86691b86a)  // subident: octave_-16
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_1_octave__16, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_1_octave__16, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(0),
                INT64_C(0xb4a24d7a84e7677b), INT64_C(0x023ff9668e89b5c4)  // subident: octave_-2
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_0_octave__2, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_0_octave__2, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(1),
                INT64_C(0xb4a24d7a84e7677b), INT64_C(0x023ff9668e89b5c4)  // subident: octave_-2
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_1_octave__2, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_1_octave__2, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(0),
                INT64_C(0x53d39c6752dac858), INT64_C(0xbcd1c5a80ab65b3e)  // subident: octave_-3
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_0_octave__3, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_0_octave__3, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(1),
                INT64_C(0x53d39c6752dac858), INT64_C(0xbcd1c5a80ab65b3e)  // subident: octave_-3
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_1_octave__3, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_1_octave__3, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(0),
                INT64_C(0xbd90d5377ba1b762), INT64_C(0xc07317d419a7548d)  // subident: octave_-4
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_0_octave__4, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_0_octave__4, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(1),
                INT64_C(0xbd90d5377ba1b762), INT64_C(0xc07317d419a7548d)  // subident: octave_-4
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_1_octave__4, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_1_octave__4, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(0),
                INT64_C(0x6d7b49e7e429850a), INT64_C(0x2e3063c622a24777)  // subident: octave_-5
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_0_octave__5, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_0_octave__5, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(1),
                INT64_C(0x6d7b49e7e429850a), INT64_C(0x2e3063c622a24777)  // subident: octave_-5
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_1_octave__5, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_1_octave__5, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(0),
                INT64_C(0xe51c98ce7d1de664), INT64_C(0x5f9478a733040c45)  // subident: octave_-6
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_0_octave__6, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_0_octave__6, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(1),
                INT64_C(0xe51c98ce7d1de664), INT64_C(0x5f9478a733040c45)  // subident: octave_-6
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_1_octave__6, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_1_octave__6, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(0),
                INT64_C(0xf11268128982754f), INT64_C(0x257a1d670430b0aa)  // subident: octave_-7
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_0_octave__7, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_0_octave__7, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(1),
                INT64_C(0xf11268128982754f), INT64_C(0x257a1d670430b0aa)  // subident: octave_-7
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_1_octave__7, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_1_octave__7, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(0),
                INT64_C(0x0ef68ec68504005e), INT64_C(0x48b6bf93a2789640)  // subident: octave_-8
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_0_octave__8, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_0_octave__8, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(1),
                INT64_C(0x0ef68ec68504005e), INT64_C(0x48b6bf93a2789640)  // subident: octave_-8
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_1_octave__8, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_1_octave__8, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(0),
                INT64_C(0x082fe255f8be6631), INT64_C(0x4e96119e22dedc81)  // subident: octave_-9
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_0_octave__9, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_0_octave__9, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }
        {
            PerlinNoiseGenerator pns;
            make_perm_table(&pns, world_seed,
                INT64_C(0xf902c0a7c9daa994), INT64_C(0x71ecd96a8da5e503), // ident: "minecraft:jagged"
                INT64_C(1),
                INT64_C(0x082fe255f8be6631), INT64_C(0x4e96119e22dedc81)  // subident: octave_-9
            );
            cudaMalloc(&d_perm_table_minecraft_jagged_1_octave__9, sizeof(PerlinNoiseGenerator));
            cudaMemcpy(d_perm_table_minecraft_jagged_1_octave__9, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);
        }

    }

    ~CudaPipeline_final_density() {
        cudaFree(d_minecraft_jagged_27_d5x132x5os16391250x0x16391250ps65565000x0x65565000_output);
        cudaFree(d_minecraft_gravel_34_d5x1x5os3278x0x3278ps13113x0x13113_output);
        cudaFree(d_minecraft_cave_layer_16_d5x1x5os13113x0x13113ps52452x0x52452_output);
        cudaFree(d_minecraft_cave_layer_52_d5x132x5os32782x32782x32782ps131130x524520x131130_output);
        cudaFree(d_minecraft_cave_layer_30_d5x1x5os6556x0x6556ps26226x0x26226_output);
        cudaFree(d_minecraft_jagged_7_d5x1x5os39339000x0x39339000ps157356000x0x157356000_output);
        cudaFree(d_minecraft_jagged_27_d5x1x5os16391250x0x16391250ps65565000x0x65565000_output);
        cudaFree(d_density_function_ShiftedNoise_3_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_cave_layer_37_d5x1x5os393x0x393ps1573x0x1573_output);
        cudaFree(d_density_function_ShiftedNoise_20_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_gravel_39_d5x1x5os524x0x524ps2098x0x2098_output);
        cudaFree(d_density_function_YClampedGradient_53_d5x132x5os65565x65565x65565ps262260x1049040x262260_output);
        cudaFree(d_minecraft_jagged_15_d5x1x5os32848065x0x32848065ps131392260x0x131392260_output);
        cudaFree(d_minecraft_cave_layer_46_d5x132x5os262260x262260x262260ps1049040x4196160x1049040_output);
        cudaFree(d_minecraft_gravel_9_d5x1x5os7212x0x7212ps28848x0x28848_output);
        cudaFree(d_minecraft_gravel_31_d5x1x5os6556x0x6556ps26226x0x26226_output);
        cudaFree(d_density_function_ShiftedNoise_11_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_jagged_49_d5x132x5os65565000x0x65565000ps262260000x0x262260000_output);
        cudaFree(d_minecraft_jagged_32_d5x132x5os32782500x0x32782500ps131130000x0x131130000_output);
        cudaFree(d_minecraft_jagged_24_d5x1x5os19669500x0x19669500ps78678000x0x78678000_output);
        cudaFree(d_minecraft_cave_layer_51_d5x132x5os163912x65565x163912ps655650x1049040x655650_output);
        cudaFree(d_minecraft_cave_layer_50_d5x132x5os327825x131130x327825ps1311300x2098080x1311300_output);
        cudaFree(d_minecraft_gravel_28_d5x1x5os13113x0x13113ps52452x0x52452_output);
        cudaFree(d_minecraft_cave_layer_8_d5x1x5os19669x0x19669ps78678x0x78678_output);
        cudaFree(d_minecraft_cave_layer_42_d5x132x5os131130x0x131130ps524520x0x524520_output);
        cudaFree(d_minecraft_gravel_1_d5x1x5os7867x0x7867ps31471x0x31471_output);
        cudaFree(d_density_function_ShiftedNoise_13_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_cave_layer_43_d5x132x5os65565x65565x65565ps262260x1049040x262260_output);
        cudaFree(d_minecraft_gravel_47_d5x132x5os1966x0x1966ps7867x0x7867_output);
        cudaFree(d_minecraft_jagged_35_d5x1x5os11473875x0x11473875ps45895500x0x45895500_output);
        cudaFree(d_density_function_ShiftedNoise_22_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_gravel_17_d5x1x5os16391x0x16391ps65565x0x65565_output);
        cudaFree(d_minecraft_cave_layer_25_d5x1x5os9834x0x9834ps39339x0x39339_output);
        cudaFree(d_density_function_ShiftedNoise_5_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_cave_layer_0_d5x1x5os16391x0x16391ps65565x0x65565_output);
        cudaFree(d_minecraft_cave_layer_45_d5x132x5os393390x393390x393390ps1573560x6294240x1573560_output);
        cudaFree(d_minecraft_jagged_32_d5x1x5os32782500x0x32782500ps131130000x0x131130000_output);
        cudaFree(d_minecraft_cave_layer_44_d5x132x5os524520x524520x524520ps2098080x8392320x2098080_output);
        cudaFree(d_minecraft_cave_layer_48_d5x132x5os655650x262260x655650ps2622600x4196160x2622600_output);
        cudaFree(d_minecraft_realism_mountains_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_density_function_Multiply_10_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_density_function_Multiply_2_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_density_function_Add_33_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_caves_barrier_1_d5x132x5os65565x65565x65565ps262260x1049040x262260_output);
        cudaFree(d_minecraft_caves_old_small_d5x132x5os65565x65565x65565ps262260x1049040x262260_output);
        cudaFree(d_density_function_Add_29_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_realism_mountains_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_realism_extreme_mountains_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_caves_barrier_2_d5x132x5os65565x65565x65565ps262260x1049040x262260_output);
        cudaFree(d_density_function_Noise_41_d5x132x5os65565x65565x65565ps262260x1049040x262260_output);
        cudaFree(d_minecraft_caves_old_medium_d5x132x5os65565x65565x65565ps262260x1049040x262260_output);
        cudaFree(d_density_function_Add_38_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_realism_extreme_mountains_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_determiner_overworld_1_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_realism_hills_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_realism_extreme_hills_factor_divergence_squared_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_realism_hills_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_determiner_cave_1_d5x132x5os65565x65565x65565ps262260x1049040x262260_output);
        cudaFree(d_minecraft_realism_extreme_hills_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_realism_extreme_hills_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_caves_old_large_d5x132x5os65565x65565x65565ps262260x1049040x262260_output);
        cudaFree(d_minecraft_realism_extreme_mountains_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_density_function_Abs_36_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_caves_new_small_d5x132x5os65565x65565x65565ps262260x1049040x262260_output);
        cudaFree(d_density_function_Multiply_19_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_caves_new_medium_d5x132x5os65565x65565x65565ps262260x1049040x262260_output);
        cudaFree(d_minecraft_realism_mountains_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_caves_new_large_d5x132x5os65565x65565x65565ps262260x1049040x262260_output);
        cudaFree(d_density_function_Multiply_18_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_realism_extreme_hills_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_realism_hills_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_new_surface_height_unprocessed_unrivered_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_new_surface_height_unprocessed_rivered_d5x1x5os65565x0x65565ps262260x0x262260_output);
        cudaFree(d_minecraft_caves_underlands_combined_d5x132x5os65565x65565x65565ps262260x1049040x262260_output);
        cudaFree(d_minecraft_new_surface_combined_processed_d5x132x5os65565x65565x65565ps262260x1049040x262260_output);
        cudaFree(d_minecraft_new_combination_unfixed_d5x132x5os65565x65565x65565ps262260x1049040x262260_output);
        cudaFree(d_minecraft_new_combination_fixed_d5x132x5os65565x65565x65565ps262260x1049040x262260_output);
        cudaFree(d_final_density_d16x2096x16os65565x65565x65565ps65565x65565x65565_output);
        cudaFree(d_perm_table_minecraft_cave_layer_0_octave__8);
        cudaFree(d_perm_table_minecraft_cave_layer_1_octave__8);
        cudaFree(d_perm_table_minecraft_gravel_0_octave__5);
        cudaFree(d_perm_table_minecraft_gravel_1_octave__5);
        cudaFree(d_perm_table_minecraft_gravel_0_octave__6);
        cudaFree(d_perm_table_minecraft_gravel_1_octave__6);
        cudaFree(d_perm_table_minecraft_gravel_0_octave__7);
        cudaFree(d_perm_table_minecraft_gravel_1_octave__7);
        cudaFree(d_perm_table_minecraft_gravel_0_octave__8);
        cudaFree(d_perm_table_minecraft_gravel_1_octave__8);
        cudaFree(d_perm_table_minecraft_jagged_0_octave__1);
        cudaFree(d_perm_table_minecraft_jagged_1_octave__1);
        cudaFree(d_perm_table_minecraft_jagged_0_octave__10);
        cudaFree(d_perm_table_minecraft_jagged_1_octave__10);
        cudaFree(d_perm_table_minecraft_jagged_0_octave__11);
        cudaFree(d_perm_table_minecraft_jagged_1_octave__11);
        cudaFree(d_perm_table_minecraft_jagged_0_octave__12);
        cudaFree(d_perm_table_minecraft_jagged_1_octave__12);
        cudaFree(d_perm_table_minecraft_jagged_0_octave__13);
        cudaFree(d_perm_table_minecraft_jagged_1_octave__13);
        cudaFree(d_perm_table_minecraft_jagged_0_octave__14);
        cudaFree(d_perm_table_minecraft_jagged_1_octave__14);
        cudaFree(d_perm_table_minecraft_jagged_0_octave__15);
        cudaFree(d_perm_table_minecraft_jagged_1_octave__15);
        cudaFree(d_perm_table_minecraft_jagged_0_octave__16);
        cudaFree(d_perm_table_minecraft_jagged_1_octave__16);
        cudaFree(d_perm_table_minecraft_jagged_0_octave__2);
        cudaFree(d_perm_table_minecraft_jagged_1_octave__2);
        cudaFree(d_perm_table_minecraft_jagged_0_octave__3);
        cudaFree(d_perm_table_minecraft_jagged_1_octave__3);
        cudaFree(d_perm_table_minecraft_jagged_0_octave__4);
        cudaFree(d_perm_table_minecraft_jagged_1_octave__4);
        cudaFree(d_perm_table_minecraft_jagged_0_octave__5);
        cudaFree(d_perm_table_minecraft_jagged_1_octave__5);
        cudaFree(d_perm_table_minecraft_jagged_0_octave__6);
        cudaFree(d_perm_table_minecraft_jagged_1_octave__6);
        cudaFree(d_perm_table_minecraft_jagged_0_octave__7);
        cudaFree(d_perm_table_minecraft_jagged_1_octave__7);
        cudaFree(d_perm_table_minecraft_jagged_0_octave__8);
        cudaFree(d_perm_table_minecraft_jagged_1_octave__8);
        cudaFree(d_perm_table_minecraft_jagged_0_octave__9);
        cudaFree(d_perm_table_minecraft_jagged_1_octave__9);
        cudaStreamDestroy(stream);
    }

    /// Execute the full density pipeline and return the target output.
    std::vector<double> run(double3 origin) {
        const int BLOCK_SIZE = 256;

        // Wave 0: minecraft_jagged_27, minecraft_gravel_34, minecraft_cave_layer_16, minecraft_cave_layer_52, minecraft_cave_layer_30, minecraft_jagged_7, minecraft_jagged_27, density_function_ShiftedNoise_3, minecraft_cave_layer_37, density_function_ShiftedNoise_20, minecraft_gravel_39, density_function_YClampedGradient_53, minecraft_jagged_15, minecraft_cave_layer_46, minecraft_gravel_9, minecraft_gravel_31, density_function_ShiftedNoise_11, minecraft_jagged_49, minecraft_jagged_32, minecraft_jagged_24, minecraft_cave_layer_51, minecraft_cave_layer_50, minecraft_gravel_28, minecraft_cave_layer_8, minecraft_cave_layer_42, minecraft_gravel_1, density_function_ShiftedNoise_13, minecraft_cave_layer_43, minecraft_gravel_47, minecraft_jagged_35, density_function_ShiftedNoise_22, minecraft_gravel_17, minecraft_cave_layer_25, density_function_ShiftedNoise_5, minecraft_cave_layer_0, minecraft_cave_layer_45, minecraft_jagged_32, minecraft_cave_layer_44, minecraft_cave_layer_48
        { // Kernel minecraft_jagged_27 with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_jagged_27<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(250, 0, 250),
                make_double3(1000, 0, 1000),
                d_perm_table_minecraft_jagged_0_octave__16,
                d_perm_table_minecraft_jagged_1_octave__16,
                d_perm_table_minecraft_jagged_0_octave__15,
                d_perm_table_minecraft_jagged_1_octave__15,
                d_perm_table_minecraft_jagged_0_octave__14,
                d_perm_table_minecraft_jagged_1_octave__14,
                d_perm_table_minecraft_jagged_0_octave__13,
                d_perm_table_minecraft_jagged_1_octave__13,
                d_perm_table_minecraft_jagged_0_octave__12,
                d_perm_table_minecraft_jagged_1_octave__12,
                d_perm_table_minecraft_jagged_0_octave__11,
                d_perm_table_minecraft_jagged_1_octave__11,
                d_perm_table_minecraft_jagged_0_octave__10,
                d_perm_table_minecraft_jagged_1_octave__10,
                d_perm_table_minecraft_jagged_0_octave__9,
                d_perm_table_minecraft_jagged_1_octave__9,
                d_perm_table_minecraft_jagged_0_octave__8,
                d_perm_table_minecraft_jagged_1_octave__8,
                d_perm_table_minecraft_jagged_0_octave__7,
                d_perm_table_minecraft_jagged_1_octave__7,
                d_perm_table_minecraft_jagged_0_octave__6,
                d_perm_table_minecraft_jagged_1_octave__6,
                d_perm_table_minecraft_jagged_0_octave__5,
                d_perm_table_minecraft_jagged_1_octave__5,
                d_perm_table_minecraft_jagged_0_octave__4,
                d_perm_table_minecraft_jagged_1_octave__4,
                d_perm_table_minecraft_jagged_0_octave__3,
                d_perm_table_minecraft_jagged_1_octave__3,
                d_perm_table_minecraft_jagged_0_octave__2,
                d_perm_table_minecraft_jagged_1_octave__2,
                d_perm_table_minecraft_jagged_0_octave__1,
                d_perm_table_minecraft_jagged_1_octave__1,
                d_minecraft_jagged_27_d5x132x5os16391250x0x16391250ps65565000x0x65565000_output
            );
        }
        { // Kernel minecraft_gravel_34 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_gravel_34<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(0.05, 0, 0.05),
                make_double3(0.2, 0, 0.2),
                d_perm_table_minecraft_gravel_0_octave__8,
                d_perm_table_minecraft_gravel_1_octave__8,
                d_perm_table_minecraft_gravel_0_octave__7,
                d_perm_table_minecraft_gravel_1_octave__7,
                d_perm_table_minecraft_gravel_0_octave__6,
                d_perm_table_minecraft_gravel_1_octave__6,
                d_perm_table_minecraft_gravel_0_octave__5,
                d_perm_table_minecraft_gravel_1_octave__5,
                d_minecraft_gravel_34_d5x1x5os3278x0x3278ps13113x0x13113_output
            );
        }
        { // Kernel minecraft_cave_layer_16 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_cave_layer_16<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(0.2, 0, 0.2),
                make_double3(0.8, 0, 0.8),
                d_perm_table_minecraft_cave_layer_0_octave__8,
                d_perm_table_minecraft_cave_layer_1_octave__8,
                d_minecraft_cave_layer_16_d5x1x5os13113x0x13113ps52452x0x52452_output
            );
        }
        { // Kernel minecraft_cave_layer_52 with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_cave_layer_52<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(0.5, 0.5, 0.5),
                make_double3(2, 8, 2),
                d_perm_table_minecraft_cave_layer_0_octave__8,
                d_perm_table_minecraft_cave_layer_1_octave__8,
                d_minecraft_cave_layer_52_d5x132x5os32782x32782x32782ps131130x524520x131130_output
            );
        }
        { // Kernel minecraft_cave_layer_30 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_cave_layer_30<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(0.1, 0, 0.1),
                make_double3(0.4, 0, 0.4),
                d_perm_table_minecraft_cave_layer_0_octave__8,
                d_perm_table_minecraft_cave_layer_1_octave__8,
                d_minecraft_cave_layer_30_d5x1x5os6556x0x6556ps26226x0x26226_output
            );
        }
        { // Kernel minecraft_jagged_7 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_jagged_7<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(600, 0, 600),
                make_double3(2400, 0, 2400),
                d_perm_table_minecraft_jagged_0_octave__16,
                d_perm_table_minecraft_jagged_1_octave__16,
                d_perm_table_minecraft_jagged_0_octave__15,
                d_perm_table_minecraft_jagged_1_octave__15,
                d_perm_table_minecraft_jagged_0_octave__14,
                d_perm_table_minecraft_jagged_1_octave__14,
                d_perm_table_minecraft_jagged_0_octave__13,
                d_perm_table_minecraft_jagged_1_octave__13,
                d_perm_table_minecraft_jagged_0_octave__12,
                d_perm_table_minecraft_jagged_1_octave__12,
                d_perm_table_minecraft_jagged_0_octave__11,
                d_perm_table_minecraft_jagged_1_octave__11,
                d_perm_table_minecraft_jagged_0_octave__10,
                d_perm_table_minecraft_jagged_1_octave__10,
                d_perm_table_minecraft_jagged_0_octave__9,
                d_perm_table_minecraft_jagged_1_octave__9,
                d_perm_table_minecraft_jagged_0_octave__8,
                d_perm_table_minecraft_jagged_1_octave__8,
                d_perm_table_minecraft_jagged_0_octave__7,
                d_perm_table_minecraft_jagged_1_octave__7,
                d_perm_table_minecraft_jagged_0_octave__6,
                d_perm_table_minecraft_jagged_1_octave__6,
                d_perm_table_minecraft_jagged_0_octave__5,
                d_perm_table_minecraft_jagged_1_octave__5,
                d_perm_table_minecraft_jagged_0_octave__4,
                d_perm_table_minecraft_jagged_1_octave__4,
                d_perm_table_minecraft_jagged_0_octave__3,
                d_perm_table_minecraft_jagged_1_octave__3,
                d_perm_table_minecraft_jagged_0_octave__2,
                d_perm_table_minecraft_jagged_1_octave__2,
                d_perm_table_minecraft_jagged_0_octave__1,
                d_perm_table_minecraft_jagged_1_octave__1,
                d_minecraft_jagged_7_d5x1x5os39339000x0x39339000ps157356000x0x157356000_output
            );
        }
        { // Kernel minecraft_jagged_27 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_jagged_27<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(250, 0, 250),
                make_double3(1000, 0, 1000),
                d_perm_table_minecraft_jagged_0_octave__16,
                d_perm_table_minecraft_jagged_1_octave__16,
                d_perm_table_minecraft_jagged_0_octave__15,
                d_perm_table_minecraft_jagged_1_octave__15,
                d_perm_table_minecraft_jagged_0_octave__14,
                d_perm_table_minecraft_jagged_1_octave__14,
                d_perm_table_minecraft_jagged_0_octave__13,
                d_perm_table_minecraft_jagged_1_octave__13,
                d_perm_table_minecraft_jagged_0_octave__12,
                d_perm_table_minecraft_jagged_1_octave__12,
                d_perm_table_minecraft_jagged_0_octave__11,
                d_perm_table_minecraft_jagged_1_octave__11,
                d_perm_table_minecraft_jagged_0_octave__10,
                d_perm_table_minecraft_jagged_1_octave__10,
                d_perm_table_minecraft_jagged_0_octave__9,
                d_perm_table_minecraft_jagged_1_octave__9,
                d_perm_table_minecraft_jagged_0_octave__8,
                d_perm_table_minecraft_jagged_1_octave__8,
                d_perm_table_minecraft_jagged_0_octave__7,
                d_perm_table_minecraft_jagged_1_octave__7,
                d_perm_table_minecraft_jagged_0_octave__6,
                d_perm_table_minecraft_jagged_1_octave__6,
                d_perm_table_minecraft_jagged_0_octave__5,
                d_perm_table_minecraft_jagged_1_octave__5,
                d_perm_table_minecraft_jagged_0_octave__4,
                d_perm_table_minecraft_jagged_1_octave__4,
                d_perm_table_minecraft_jagged_0_octave__3,
                d_perm_table_minecraft_jagged_1_octave__3,
                d_perm_table_minecraft_jagged_0_octave__2,
                d_perm_table_minecraft_jagged_1_octave__2,
                d_perm_table_minecraft_jagged_0_octave__1,
                d_perm_table_minecraft_jagged_1_octave__1,
                d_minecraft_jagged_27_d5x1x5os16391250x0x16391250ps65565000x0x65565000_output
            );
        }
        { // Kernel density_function_ShiftedNoise_3 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            density_function_ShiftedNoise_3<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_perm_table_minecraft_gravel_0_octave__8,
                d_perm_table_minecraft_gravel_1_octave__8,
                d_perm_table_minecraft_gravel_0_octave__7,
                d_perm_table_minecraft_gravel_1_octave__7,
                d_perm_table_minecraft_gravel_0_octave__6,
                d_perm_table_minecraft_gravel_1_octave__6,
                d_perm_table_minecraft_gravel_0_octave__5,
                d_perm_table_minecraft_gravel_1_octave__5,
                d_density_function_ShiftedNoise_3_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_cave_layer_37 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_cave_layer_37<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(0.006, 0, 0.006),
                make_double3(0.024, 0, 0.024),
                d_perm_table_minecraft_cave_layer_0_octave__8,
                d_perm_table_minecraft_cave_layer_1_octave__8,
                d_minecraft_cave_layer_37_d5x1x5os393x0x393ps1573x0x1573_output
            );
        }
        { // Kernel density_function_ShiftedNoise_20 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            density_function_ShiftedNoise_20<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_perm_table_minecraft_gravel_0_octave__8,
                d_perm_table_minecraft_gravel_1_octave__8,
                d_perm_table_minecraft_gravel_0_octave__7,
                d_perm_table_minecraft_gravel_1_octave__7,
                d_perm_table_minecraft_gravel_0_octave__6,
                d_perm_table_minecraft_gravel_1_octave__6,
                d_perm_table_minecraft_gravel_0_octave__5,
                d_perm_table_minecraft_gravel_1_octave__5,
                d_density_function_ShiftedNoise_20_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_gravel_39 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_gravel_39<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(0.008, 0, 0.008),
                make_double3(0.032, 0, 0.032),
                d_perm_table_minecraft_gravel_0_octave__8,
                d_perm_table_minecraft_gravel_1_octave__8,
                d_perm_table_minecraft_gravel_0_octave__7,
                d_perm_table_minecraft_gravel_1_octave__7,
                d_perm_table_minecraft_gravel_0_octave__6,
                d_perm_table_minecraft_gravel_1_octave__6,
                d_perm_table_minecraft_gravel_0_octave__5,
                d_perm_table_minecraft_gravel_1_octave__5,
                d_minecraft_gravel_39_d5x1x5os524x0x524ps2098x0x2098_output
            );
        }
        { // Kernel density_function_YClampedGradient_53 with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            density_function_YClampedGradient_53<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(1, 1, 1),
                make_double3(4, 16, 4),
                d_density_function_YClampedGradient_53_d5x132x5os65565x65565x65565ps262260x1049040x262260_output
            );
        }
        { // Kernel minecraft_jagged_15 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_jagged_15<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(501, 0, 501),
                make_double3(2004, 0, 2004),
                d_perm_table_minecraft_jagged_0_octave__16,
                d_perm_table_minecraft_jagged_1_octave__16,
                d_perm_table_minecraft_jagged_0_octave__15,
                d_perm_table_minecraft_jagged_1_octave__15,
                d_perm_table_minecraft_jagged_0_octave__14,
                d_perm_table_minecraft_jagged_1_octave__14,
                d_perm_table_minecraft_jagged_0_octave__13,
                d_perm_table_minecraft_jagged_1_octave__13,
                d_perm_table_minecraft_jagged_0_octave__12,
                d_perm_table_minecraft_jagged_1_octave__12,
                d_perm_table_minecraft_jagged_0_octave__11,
                d_perm_table_minecraft_jagged_1_octave__11,
                d_perm_table_minecraft_jagged_0_octave__10,
                d_perm_table_minecraft_jagged_1_octave__10,
                d_perm_table_minecraft_jagged_0_octave__9,
                d_perm_table_minecraft_jagged_1_octave__9,
                d_perm_table_minecraft_jagged_0_octave__8,
                d_perm_table_minecraft_jagged_1_octave__8,
                d_perm_table_minecraft_jagged_0_octave__7,
                d_perm_table_minecraft_jagged_1_octave__7,
                d_perm_table_minecraft_jagged_0_octave__6,
                d_perm_table_minecraft_jagged_1_octave__6,
                d_perm_table_minecraft_jagged_0_octave__5,
                d_perm_table_minecraft_jagged_1_octave__5,
                d_perm_table_minecraft_jagged_0_octave__4,
                d_perm_table_minecraft_jagged_1_octave__4,
                d_perm_table_minecraft_jagged_0_octave__3,
                d_perm_table_minecraft_jagged_1_octave__3,
                d_perm_table_minecraft_jagged_0_octave__2,
                d_perm_table_minecraft_jagged_1_octave__2,
                d_perm_table_minecraft_jagged_0_octave__1,
                d_perm_table_minecraft_jagged_1_octave__1,
                d_minecraft_jagged_15_d5x1x5os32848065x0x32848065ps131392260x0x131392260_output
            );
        }
        { // Kernel minecraft_cave_layer_46 with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_cave_layer_46<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(4, 4, 4),
                make_double3(16, 64, 16),
                d_perm_table_minecraft_cave_layer_0_octave__8,
                d_perm_table_minecraft_cave_layer_1_octave__8,
                d_minecraft_cave_layer_46_d5x132x5os262260x262260x262260ps1049040x4196160x1049040_output
            );
        }
        { // Kernel minecraft_gravel_9 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_gravel_9<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(0.11, 0, 0.11),
                make_double3(0.44, 0, 0.44),
                d_perm_table_minecraft_gravel_0_octave__8,
                d_perm_table_minecraft_gravel_1_octave__8,
                d_perm_table_minecraft_gravel_0_octave__7,
                d_perm_table_minecraft_gravel_1_octave__7,
                d_perm_table_minecraft_gravel_0_octave__6,
                d_perm_table_minecraft_gravel_1_octave__6,
                d_perm_table_minecraft_gravel_0_octave__5,
                d_perm_table_minecraft_gravel_1_octave__5,
                d_minecraft_gravel_9_d5x1x5os7212x0x7212ps28848x0x28848_output
            );
        }
        { // Kernel minecraft_gravel_31 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_gravel_31<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(0.1, 0, 0.1),
                make_double3(0.4, 0, 0.4),
                d_perm_table_minecraft_gravel_0_octave__8,
                d_perm_table_minecraft_gravel_1_octave__8,
                d_perm_table_minecraft_gravel_0_octave__7,
                d_perm_table_minecraft_gravel_1_octave__7,
                d_perm_table_minecraft_gravel_0_octave__6,
                d_perm_table_minecraft_gravel_1_octave__6,
                d_perm_table_minecraft_gravel_0_octave__5,
                d_perm_table_minecraft_gravel_1_octave__5,
                d_minecraft_gravel_31_d5x1x5os6556x0x6556ps26226x0x26226_output
            );
        }
        { // Kernel density_function_ShiftedNoise_11 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            density_function_ShiftedNoise_11<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_perm_table_minecraft_gravel_0_octave__8,
                d_perm_table_minecraft_gravel_1_octave__8,
                d_perm_table_minecraft_gravel_0_octave__7,
                d_perm_table_minecraft_gravel_1_octave__7,
                d_perm_table_minecraft_gravel_0_octave__6,
                d_perm_table_minecraft_gravel_1_octave__6,
                d_perm_table_minecraft_gravel_0_octave__5,
                d_perm_table_minecraft_gravel_1_octave__5,
                d_density_function_ShiftedNoise_11_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_jagged_49 with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_jagged_49<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(1000, 0, 1000),
                make_double3(4000, 0, 4000),
                d_perm_table_minecraft_jagged_0_octave__16,
                d_perm_table_minecraft_jagged_1_octave__16,
                d_perm_table_minecraft_jagged_0_octave__15,
                d_perm_table_minecraft_jagged_1_octave__15,
                d_perm_table_minecraft_jagged_0_octave__14,
                d_perm_table_minecraft_jagged_1_octave__14,
                d_perm_table_minecraft_jagged_0_octave__13,
                d_perm_table_minecraft_jagged_1_octave__13,
                d_perm_table_minecraft_jagged_0_octave__12,
                d_perm_table_minecraft_jagged_1_octave__12,
                d_perm_table_minecraft_jagged_0_octave__11,
                d_perm_table_minecraft_jagged_1_octave__11,
                d_perm_table_minecraft_jagged_0_octave__10,
                d_perm_table_minecraft_jagged_1_octave__10,
                d_perm_table_minecraft_jagged_0_octave__9,
                d_perm_table_minecraft_jagged_1_octave__9,
                d_perm_table_minecraft_jagged_0_octave__8,
                d_perm_table_minecraft_jagged_1_octave__8,
                d_perm_table_minecraft_jagged_0_octave__7,
                d_perm_table_minecraft_jagged_1_octave__7,
                d_perm_table_minecraft_jagged_0_octave__6,
                d_perm_table_minecraft_jagged_1_octave__6,
                d_perm_table_minecraft_jagged_0_octave__5,
                d_perm_table_minecraft_jagged_1_octave__5,
                d_perm_table_minecraft_jagged_0_octave__4,
                d_perm_table_minecraft_jagged_1_octave__4,
                d_perm_table_minecraft_jagged_0_octave__3,
                d_perm_table_minecraft_jagged_1_octave__3,
                d_perm_table_minecraft_jagged_0_octave__2,
                d_perm_table_minecraft_jagged_1_octave__2,
                d_perm_table_minecraft_jagged_0_octave__1,
                d_perm_table_minecraft_jagged_1_octave__1,
                d_minecraft_jagged_49_d5x132x5os65565000x0x65565000ps262260000x0x262260000_output
            );
        }
        { // Kernel minecraft_jagged_32 with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_jagged_32<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(500, 0, 500),
                make_double3(2000, 0, 2000),
                d_perm_table_minecraft_jagged_0_octave__16,
                d_perm_table_minecraft_jagged_1_octave__16,
                d_perm_table_minecraft_jagged_0_octave__15,
                d_perm_table_minecraft_jagged_1_octave__15,
                d_perm_table_minecraft_jagged_0_octave__14,
                d_perm_table_minecraft_jagged_1_octave__14,
                d_perm_table_minecraft_jagged_0_octave__13,
                d_perm_table_minecraft_jagged_1_octave__13,
                d_perm_table_minecraft_jagged_0_octave__12,
                d_perm_table_minecraft_jagged_1_octave__12,
                d_perm_table_minecraft_jagged_0_octave__11,
                d_perm_table_minecraft_jagged_1_octave__11,
                d_perm_table_minecraft_jagged_0_octave__10,
                d_perm_table_minecraft_jagged_1_octave__10,
                d_perm_table_minecraft_jagged_0_octave__9,
                d_perm_table_minecraft_jagged_1_octave__9,
                d_perm_table_minecraft_jagged_0_octave__8,
                d_perm_table_minecraft_jagged_1_octave__8,
                d_perm_table_minecraft_jagged_0_octave__7,
                d_perm_table_minecraft_jagged_1_octave__7,
                d_perm_table_minecraft_jagged_0_octave__6,
                d_perm_table_minecraft_jagged_1_octave__6,
                d_perm_table_minecraft_jagged_0_octave__5,
                d_perm_table_minecraft_jagged_1_octave__5,
                d_perm_table_minecraft_jagged_0_octave__4,
                d_perm_table_minecraft_jagged_1_octave__4,
                d_perm_table_minecraft_jagged_0_octave__3,
                d_perm_table_minecraft_jagged_1_octave__3,
                d_perm_table_minecraft_jagged_0_octave__2,
                d_perm_table_minecraft_jagged_1_octave__2,
                d_perm_table_minecraft_jagged_0_octave__1,
                d_perm_table_minecraft_jagged_1_octave__1,
                d_minecraft_jagged_32_d5x132x5os32782500x0x32782500ps131130000x0x131130000_output
            );
        }
        { // Kernel minecraft_jagged_24 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_jagged_24<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(300, 0, 300),
                make_double3(1200, 0, 1200),
                d_perm_table_minecraft_jagged_0_octave__16,
                d_perm_table_minecraft_jagged_1_octave__16,
                d_perm_table_minecraft_jagged_0_octave__15,
                d_perm_table_minecraft_jagged_1_octave__15,
                d_perm_table_minecraft_jagged_0_octave__14,
                d_perm_table_minecraft_jagged_1_octave__14,
                d_perm_table_minecraft_jagged_0_octave__13,
                d_perm_table_minecraft_jagged_1_octave__13,
                d_perm_table_minecraft_jagged_0_octave__12,
                d_perm_table_minecraft_jagged_1_octave__12,
                d_perm_table_minecraft_jagged_0_octave__11,
                d_perm_table_minecraft_jagged_1_octave__11,
                d_perm_table_minecraft_jagged_0_octave__10,
                d_perm_table_minecraft_jagged_1_octave__10,
                d_perm_table_minecraft_jagged_0_octave__9,
                d_perm_table_minecraft_jagged_1_octave__9,
                d_perm_table_minecraft_jagged_0_octave__8,
                d_perm_table_minecraft_jagged_1_octave__8,
                d_perm_table_minecraft_jagged_0_octave__7,
                d_perm_table_minecraft_jagged_1_octave__7,
                d_perm_table_minecraft_jagged_0_octave__6,
                d_perm_table_minecraft_jagged_1_octave__6,
                d_perm_table_minecraft_jagged_0_octave__5,
                d_perm_table_minecraft_jagged_1_octave__5,
                d_perm_table_minecraft_jagged_0_octave__4,
                d_perm_table_minecraft_jagged_1_octave__4,
                d_perm_table_minecraft_jagged_0_octave__3,
                d_perm_table_minecraft_jagged_1_octave__3,
                d_perm_table_minecraft_jagged_0_octave__2,
                d_perm_table_minecraft_jagged_1_octave__2,
                d_perm_table_minecraft_jagged_0_octave__1,
                d_perm_table_minecraft_jagged_1_octave__1,
                d_minecraft_jagged_24_d5x1x5os19669500x0x19669500ps78678000x0x78678000_output
            );
        }
        { // Kernel minecraft_cave_layer_51 with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_cave_layer_51<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(2.5, 1, 2.5),
                make_double3(10, 16, 10),
                d_perm_table_minecraft_cave_layer_0_octave__8,
                d_perm_table_minecraft_cave_layer_1_octave__8,
                d_minecraft_cave_layer_51_d5x132x5os163912x65565x163912ps655650x1049040x655650_output
            );
        }
        { // Kernel minecraft_cave_layer_50 with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_cave_layer_50<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(5, 2, 5),
                make_double3(20, 32, 20),
                d_perm_table_minecraft_cave_layer_0_octave__8,
                d_perm_table_minecraft_cave_layer_1_octave__8,
                d_minecraft_cave_layer_50_d5x132x5os327825x131130x327825ps1311300x2098080x1311300_output
            );
        }
        { // Kernel minecraft_gravel_28 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_gravel_28<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(0.2, 0, 0.2),
                make_double3(0.8, 0, 0.8),
                d_perm_table_minecraft_gravel_0_octave__8,
                d_perm_table_minecraft_gravel_1_octave__8,
                d_perm_table_minecraft_gravel_0_octave__7,
                d_perm_table_minecraft_gravel_1_octave__7,
                d_perm_table_minecraft_gravel_0_octave__6,
                d_perm_table_minecraft_gravel_1_octave__6,
                d_perm_table_minecraft_gravel_0_octave__5,
                d_perm_table_minecraft_gravel_1_octave__5,
                d_minecraft_gravel_28_d5x1x5os13113x0x13113ps52452x0x52452_output
            );
        }
        { // Kernel minecraft_cave_layer_8 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_cave_layer_8<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(0.3, 0, 0.3),
                make_double3(1.2, 0, 1.2),
                d_perm_table_minecraft_cave_layer_0_octave__8,
                d_perm_table_minecraft_cave_layer_1_octave__8,
                d_minecraft_cave_layer_8_d5x1x5os19669x0x19669ps78678x0x78678_output
            );
        }
        { // Kernel minecraft_cave_layer_42 with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_cave_layer_42<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(2, 0, 2),
                make_double3(8, 0, 8),
                d_perm_table_minecraft_cave_layer_0_octave__8,
                d_perm_table_minecraft_cave_layer_1_octave__8,
                d_minecraft_cave_layer_42_d5x132x5os131130x0x131130ps524520x0x524520_output
            );
        }
        { // Kernel minecraft_gravel_1 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_gravel_1<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(0.12, 0, 0.12),
                make_double3(0.48, 0, 0.48),
                d_perm_table_minecraft_gravel_0_octave__8,
                d_perm_table_minecraft_gravel_1_octave__8,
                d_perm_table_minecraft_gravel_0_octave__7,
                d_perm_table_minecraft_gravel_1_octave__7,
                d_perm_table_minecraft_gravel_0_octave__6,
                d_perm_table_minecraft_gravel_1_octave__6,
                d_perm_table_minecraft_gravel_0_octave__5,
                d_perm_table_minecraft_gravel_1_octave__5,
                d_minecraft_gravel_1_d5x1x5os7867x0x7867ps31471x0x31471_output
            );
        }
        { // Kernel density_function_ShiftedNoise_13 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            density_function_ShiftedNoise_13<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_perm_table_minecraft_gravel_0_octave__8,
                d_perm_table_minecraft_gravel_1_octave__8,
                d_perm_table_minecraft_gravel_0_octave__7,
                d_perm_table_minecraft_gravel_1_octave__7,
                d_perm_table_minecraft_gravel_0_octave__6,
                d_perm_table_minecraft_gravel_1_octave__6,
                d_perm_table_minecraft_gravel_0_octave__5,
                d_perm_table_minecraft_gravel_1_octave__5,
                d_density_function_ShiftedNoise_13_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_cave_layer_43 with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_cave_layer_43<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(1, 1, 1),
                make_double3(4, 16, 4),
                d_perm_table_minecraft_cave_layer_0_octave__8,
                d_perm_table_minecraft_cave_layer_1_octave__8,
                d_minecraft_cave_layer_43_d5x132x5os65565x65565x65565ps262260x1049040x262260_output
            );
        }
        { // Kernel minecraft_gravel_47 with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_gravel_47<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(0.03, 0, 0.03),
                make_double3(0.12, 0, 0.12),
                d_perm_table_minecraft_gravel_0_octave__8,
                d_perm_table_minecraft_gravel_1_octave__8,
                d_perm_table_minecraft_gravel_0_octave__7,
                d_perm_table_minecraft_gravel_1_octave__7,
                d_perm_table_minecraft_gravel_0_octave__6,
                d_perm_table_minecraft_gravel_1_octave__6,
                d_perm_table_minecraft_gravel_0_octave__5,
                d_perm_table_minecraft_gravel_1_octave__5,
                d_minecraft_gravel_47_d5x132x5os1966x0x1966ps7867x0x7867_output
            );
        }
        { // Kernel minecraft_jagged_35 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_jagged_35<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(175, 0, 175),
                make_double3(700, 0, 700),
                d_perm_table_minecraft_jagged_0_octave__16,
                d_perm_table_minecraft_jagged_1_octave__16,
                d_perm_table_minecraft_jagged_0_octave__15,
                d_perm_table_minecraft_jagged_1_octave__15,
                d_perm_table_minecraft_jagged_0_octave__14,
                d_perm_table_minecraft_jagged_1_octave__14,
                d_perm_table_minecraft_jagged_0_octave__13,
                d_perm_table_minecraft_jagged_1_octave__13,
                d_perm_table_minecraft_jagged_0_octave__12,
                d_perm_table_minecraft_jagged_1_octave__12,
                d_perm_table_minecraft_jagged_0_octave__11,
                d_perm_table_minecraft_jagged_1_octave__11,
                d_perm_table_minecraft_jagged_0_octave__10,
                d_perm_table_minecraft_jagged_1_octave__10,
                d_perm_table_minecraft_jagged_0_octave__9,
                d_perm_table_minecraft_jagged_1_octave__9,
                d_perm_table_minecraft_jagged_0_octave__8,
                d_perm_table_minecraft_jagged_1_octave__8,
                d_perm_table_minecraft_jagged_0_octave__7,
                d_perm_table_minecraft_jagged_1_octave__7,
                d_perm_table_minecraft_jagged_0_octave__6,
                d_perm_table_minecraft_jagged_1_octave__6,
                d_perm_table_minecraft_jagged_0_octave__5,
                d_perm_table_minecraft_jagged_1_octave__5,
                d_perm_table_minecraft_jagged_0_octave__4,
                d_perm_table_minecraft_jagged_1_octave__4,
                d_perm_table_minecraft_jagged_0_octave__3,
                d_perm_table_minecraft_jagged_1_octave__3,
                d_perm_table_minecraft_jagged_0_octave__2,
                d_perm_table_minecraft_jagged_1_octave__2,
                d_perm_table_minecraft_jagged_0_octave__1,
                d_perm_table_minecraft_jagged_1_octave__1,
                d_minecraft_jagged_35_d5x1x5os11473875x0x11473875ps45895500x0x45895500_output
            );
        }
        { // Kernel density_function_ShiftedNoise_22 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            density_function_ShiftedNoise_22<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_perm_table_minecraft_gravel_0_octave__8,
                d_perm_table_minecraft_gravel_1_octave__8,
                d_perm_table_minecraft_gravel_0_octave__7,
                d_perm_table_minecraft_gravel_1_octave__7,
                d_perm_table_minecraft_gravel_0_octave__6,
                d_perm_table_minecraft_gravel_1_octave__6,
                d_perm_table_minecraft_gravel_0_octave__5,
                d_perm_table_minecraft_gravel_1_octave__5,
                d_density_function_ShiftedNoise_22_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_gravel_17 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_gravel_17<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(0.25, 0, 0.25),
                make_double3(1, 0, 1),
                d_perm_table_minecraft_gravel_0_octave__8,
                d_perm_table_minecraft_gravel_1_octave__8,
                d_perm_table_minecraft_gravel_0_octave__7,
                d_perm_table_minecraft_gravel_1_octave__7,
                d_perm_table_minecraft_gravel_0_octave__6,
                d_perm_table_minecraft_gravel_1_octave__6,
                d_perm_table_minecraft_gravel_0_octave__5,
                d_perm_table_minecraft_gravel_1_octave__5,
                d_minecraft_gravel_17_d5x1x5os16391x0x16391ps65565x0x65565_output
            );
        }
        { // Kernel minecraft_cave_layer_25 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_cave_layer_25<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(0.15, 0, 0.15),
                make_double3(0.6, 0, 0.6),
                d_perm_table_minecraft_cave_layer_0_octave__8,
                d_perm_table_minecraft_cave_layer_1_octave__8,
                d_minecraft_cave_layer_25_d5x1x5os9834x0x9834ps39339x0x39339_output
            );
        }
        { // Kernel density_function_ShiftedNoise_5 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            density_function_ShiftedNoise_5<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_perm_table_minecraft_gravel_0_octave__8,
                d_perm_table_minecraft_gravel_1_octave__8,
                d_perm_table_minecraft_gravel_0_octave__7,
                d_perm_table_minecraft_gravel_1_octave__7,
                d_perm_table_minecraft_gravel_0_octave__6,
                d_perm_table_minecraft_gravel_1_octave__6,
                d_perm_table_minecraft_gravel_0_octave__5,
                d_perm_table_minecraft_gravel_1_octave__5,
                d_density_function_ShiftedNoise_5_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_cave_layer_0 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_cave_layer_0<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(0.25, 0, 0.25),
                make_double3(1, 0, 1),
                d_perm_table_minecraft_cave_layer_0_octave__8,
                d_perm_table_minecraft_cave_layer_1_octave__8,
                d_minecraft_cave_layer_0_d5x1x5os16391x0x16391ps65565x0x65565_output
            );
        }
        { // Kernel minecraft_cave_layer_45 with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_cave_layer_45<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(6, 6, 6),
                make_double3(24, 96, 24),
                d_perm_table_minecraft_cave_layer_0_octave__8,
                d_perm_table_minecraft_cave_layer_1_octave__8,
                d_minecraft_cave_layer_45_d5x132x5os393390x393390x393390ps1573560x6294240x1573560_output
            );
        }
        { // Kernel minecraft_jagged_32 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_jagged_32<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(500, 0, 500),
                make_double3(2000, 0, 2000),
                d_perm_table_minecraft_jagged_0_octave__16,
                d_perm_table_minecraft_jagged_1_octave__16,
                d_perm_table_minecraft_jagged_0_octave__15,
                d_perm_table_minecraft_jagged_1_octave__15,
                d_perm_table_minecraft_jagged_0_octave__14,
                d_perm_table_minecraft_jagged_1_octave__14,
                d_perm_table_minecraft_jagged_0_octave__13,
                d_perm_table_minecraft_jagged_1_octave__13,
                d_perm_table_minecraft_jagged_0_octave__12,
                d_perm_table_minecraft_jagged_1_octave__12,
                d_perm_table_minecraft_jagged_0_octave__11,
                d_perm_table_minecraft_jagged_1_octave__11,
                d_perm_table_minecraft_jagged_0_octave__10,
                d_perm_table_minecraft_jagged_1_octave__10,
                d_perm_table_minecraft_jagged_0_octave__9,
                d_perm_table_minecraft_jagged_1_octave__9,
                d_perm_table_minecraft_jagged_0_octave__8,
                d_perm_table_minecraft_jagged_1_octave__8,
                d_perm_table_minecraft_jagged_0_octave__7,
                d_perm_table_minecraft_jagged_1_octave__7,
                d_perm_table_minecraft_jagged_0_octave__6,
                d_perm_table_minecraft_jagged_1_octave__6,
                d_perm_table_minecraft_jagged_0_octave__5,
                d_perm_table_minecraft_jagged_1_octave__5,
                d_perm_table_minecraft_jagged_0_octave__4,
                d_perm_table_minecraft_jagged_1_octave__4,
                d_perm_table_minecraft_jagged_0_octave__3,
                d_perm_table_minecraft_jagged_1_octave__3,
                d_perm_table_minecraft_jagged_0_octave__2,
                d_perm_table_minecraft_jagged_1_octave__2,
                d_perm_table_minecraft_jagged_0_octave__1,
                d_perm_table_minecraft_jagged_1_octave__1,
                d_minecraft_jagged_32_d5x1x5os32782500x0x32782500ps131130000x0x131130000_output
            );
        }
        { // Kernel minecraft_cave_layer_44 with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_cave_layer_44<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(8, 8, 8),
                make_double3(32, 128, 32),
                d_perm_table_minecraft_cave_layer_0_octave__8,
                d_perm_table_minecraft_cave_layer_1_octave__8,
                d_minecraft_cave_layer_44_d5x132x5os524520x524520x524520ps2098080x8392320x2098080_output
            );
        }
        { // Kernel minecraft_cave_layer_48 with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_cave_layer_48<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(10, 4, 10),
                make_double3(40, 64, 40),
                d_perm_table_minecraft_cave_layer_0_octave__8,
                d_perm_table_minecraft_cave_layer_1_octave__8,
                d_minecraft_cave_layer_48_d5x132x5os655650x262260x655650ps2622600x4196160x2622600_output
            );
        }
        cudaStreamSynchronize(stream);

        // Wave 1: minecraft_realism_mountains_factor_original, density_function_Multiply_10, density_function_Multiply_2, density_function_Add_33, minecraft_caves_barrier_1, minecraft_caves_old_small, density_function_Add_29, minecraft_realism_mountains_jagged_original, minecraft_realism_extreme_mountains_factor_original, minecraft_caves_barrier_2, density_function_Noise_41, minecraft_caves_old_medium, density_function_Add_38, minecraft_realism_extreme_mountains_jagged_original, minecraft_determiner_overworld_1, minecraft_realism_hills_jagged_original, minecraft_realism_extreme_hills_factor_divergence_squared, minecraft_realism_hills_factor_original, minecraft_determiner_cave_1, minecraft_realism_extreme_hills_factor_original, minecraft_realism_extreme_hills_jagged_original, minecraft_caves_old_large
        { // Kernel minecraft_realism_mountains_factor_original with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_realism_mountains_factor_original<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_gravel_9_d5x1x5os7212x0x7212ps28848x0x28848_output,
                d_minecraft_realism_mountains_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel density_function_Multiply_10 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            density_function_Multiply_10<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_density_function_ShiftedNoise_11_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_gravel_9_d5x1x5os7212x0x7212ps28848x0x28848_output,
                d_density_function_ShiftedNoise_13_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_density_function_Multiply_10_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel density_function_Multiply_2 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            density_function_Multiply_2<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_density_function_ShiftedNoise_3_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_gravel_1_d5x1x5os7867x0x7867ps31471x0x31471_output,
                d_density_function_ShiftedNoise_5_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_density_function_Multiply_2_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel density_function_Add_33 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            density_function_Add_33<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_gravel_34_d5x1x5os3278x0x3278ps13113x0x13113_output,
                d_minecraft_cave_layer_16_d5x1x5os13113x0x13113ps52452x0x52452_output,
                d_minecraft_jagged_35_d5x1x5os11473875x0x11473875ps45895500x0x45895500_output,
                d_density_function_Add_33_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_caves_barrier_1 with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_caves_barrier_1<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(1, 1, 1),
                make_double3(4, 16, 4),
                d_minecraft_cave_layer_52_d5x132x5os32782x32782x32782ps131130x524520x131130_output,
                d_minecraft_caves_barrier_1_d5x132x5os65565x65565x65565ps262260x1049040x262260_output
            );
        }
        { // Kernel minecraft_caves_old_small with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_caves_old_small<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(1, 1, 1),
                make_double3(4, 16, 4),
                d_minecraft_cave_layer_44_d5x132x5os524520x524520x524520ps2098080x8392320x2098080_output,
                d_minecraft_caves_old_small_d5x132x5os65565x65565x65565ps262260x1049040x262260_output
            );
        }
        { // Kernel density_function_Add_29 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            density_function_Add_29<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_gravel_31_d5x1x5os6556x0x6556ps26226x0x26226_output,
                d_minecraft_cave_layer_30_d5x1x5os6556x0x6556ps26226x0x26226_output,
                d_minecraft_jagged_32_d5x1x5os32782500x0x32782500ps131130000x0x131130000_output,
                d_density_function_Add_29_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_realism_mountains_jagged_original with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_realism_mountains_jagged_original<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_jagged_15_d5x1x5os32848065x0x32848065ps131392260x0x131392260_output,
                d_minecraft_realism_mountains_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_realism_extreme_mountains_factor_original with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_realism_extreme_mountains_factor_original<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_gravel_1_d5x1x5os7867x0x7867ps31471x0x31471_output,
                d_minecraft_realism_extreme_mountains_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_caves_barrier_2 with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_caves_barrier_2<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(1, 1, 1),
                make_double3(4, 16, 4),
                d_minecraft_cave_layer_43_d5x132x5os65565x65565x65565ps262260x1049040x262260_output,
                d_minecraft_caves_barrier_2_d5x132x5os65565x65565x65565ps262260x1049040x262260_output
            );
        }
        { // Kernel density_function_Noise_41 with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            density_function_Noise_41<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(1, 1, 1),
                make_double3(4, 16, 4),
                d_minecraft_cave_layer_42_d5x132x5os131130x0x131130ps524520x0x524520_output,
                d_density_function_Noise_41_d5x132x5os65565x65565x65565ps262260x1049040x262260_output
            );
        }
        { // Kernel minecraft_caves_old_medium with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_caves_old_medium<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(1, 1, 1),
                make_double3(4, 16, 4),
                d_minecraft_cave_layer_45_d5x132x5os393390x393390x393390ps1573560x6294240x1573560_output,
                d_minecraft_caves_old_medium_d5x132x5os65565x65565x65565ps262260x1049040x262260_output
            );
        }
        { // Kernel density_function_Add_38 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            density_function_Add_38<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_gravel_39_d5x1x5os524x0x524ps2098x0x2098_output,
                d_minecraft_cave_layer_37_d5x1x5os393x0x393ps1573x0x1573_output,
                d_density_function_Add_38_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_realism_extreme_mountains_jagged_original with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_realism_extreme_mountains_jagged_original<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_jagged_7_d5x1x5os39339000x0x39339000ps157356000x0x157356000_output,
                d_minecraft_realism_extreme_mountains_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_determiner_overworld_1 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_determiner_overworld_1<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_cave_layer_37_d5x1x5os393x0x393ps1573x0x1573_output,
                d_minecraft_determiner_overworld_1_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_realism_hills_jagged_original with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_realism_hills_jagged_original<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_jagged_27_d5x1x5os16391250x0x16391250ps65565000x0x65565000_output,
                d_minecraft_realism_hills_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_realism_extreme_hills_factor_divergence_squared with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_realism_extreme_hills_factor_divergence_squared<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_density_function_ShiftedNoise_20_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_gravel_17_d5x1x5os16391x0x16391ps65565x0x65565_output,
                d_density_function_ShiftedNoise_22_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_realism_extreme_hills_factor_divergence_squared_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_realism_hills_factor_original with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_realism_hills_factor_original<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_gravel_28_d5x1x5os13113x0x13113ps52452x0x52452_output,
                d_minecraft_realism_hills_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_determiner_cave_1 with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_determiner_cave_1<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(1, 1, 1),
                make_double3(4, 16, 4),
                d_minecraft_gravel_47_d5x132x5os1966x0x1966ps7867x0x7867_output,
                d_minecraft_determiner_cave_1_d5x132x5os65565x65565x65565ps262260x1049040x262260_output
            );
        }
        { // Kernel minecraft_realism_extreme_hills_factor_original with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_realism_extreme_hills_factor_original<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_gravel_17_d5x1x5os16391x0x16391ps65565x0x65565_output,
                d_minecraft_realism_extreme_hills_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_realism_extreme_hills_jagged_original with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_realism_extreme_hills_jagged_original<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_jagged_24_d5x1x5os19669500x0x19669500ps78678000x0x78678000_output,
                d_minecraft_realism_extreme_hills_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_caves_old_large with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_caves_old_large<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(1, 1, 1),
                make_double3(4, 16, 4),
                d_minecraft_cave_layer_46_d5x132x5os262260x262260x262260ps1049040x4196160x1049040_output,
                d_minecraft_caves_old_large_d5x132x5os65565x65565x65565ps262260x1049040x262260_output
            );
        }
        cudaStreamSynchronize(stream);

        // Wave 2: minecraft_realism_extreme_mountains_slope_height, density_function_Abs_36, minecraft_caves_new_small, density_function_Multiply_19, minecraft_caves_new_medium, minecraft_realism_mountains_slope_height, minecraft_caves_new_large
        { // Kernel minecraft_realism_extreme_mountains_slope_height with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_realism_extreme_mountains_slope_height<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_gravel_1_d5x1x5os7867x0x7867ps31471x0x31471_output,
                d_density_function_Multiply_2_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_realism_extreme_mountains_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_realism_extreme_mountains_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_cave_layer_0_d5x1x5os16391x0x16391ps65565x0x65565_output,
                d_minecraft_realism_extreme_mountains_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel density_function_Abs_36 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            density_function_Abs_36<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_determiner_overworld_1_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_density_function_Add_38_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_density_function_Abs_36_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_caves_new_small with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_caves_new_small<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(1, 1, 1),
                make_double3(4, 16, 4),
                d_minecraft_determiner_cave_1_d5x132x5os65565x65565x65565ps262260x1049040x262260_output,
                d_minecraft_cave_layer_48_d5x132x5os655650x262260x655650ps2622600x4196160x2622600_output,
                d_minecraft_jagged_49_d5x132x5os65565000x0x65565000ps262260000x0x262260000_output,
                d_minecraft_caves_new_small_d5x132x5os65565x65565x65565ps262260x1049040x262260_output
            );
        }
        { // Kernel density_function_Multiply_19 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            density_function_Multiply_19<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_realism_extreme_hills_factor_divergence_squared_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_density_function_Multiply_19_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_caves_new_medium with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_caves_new_medium<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(1, 1, 1),
                make_double3(4, 16, 4),
                d_minecraft_determiner_cave_1_d5x132x5os65565x65565x65565ps262260x1049040x262260_output,
                d_minecraft_cave_layer_50_d5x132x5os327825x131130x327825ps1311300x2098080x1311300_output,
                d_minecraft_jagged_32_d5x132x5os32782500x0x32782500ps131130000x0x131130000_output,
                d_minecraft_caves_new_medium_d5x132x5os65565x65565x65565ps262260x1049040x262260_output
            );
        }
        { // Kernel minecraft_realism_mountains_slope_height with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_realism_mountains_slope_height<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_gravel_9_d5x1x5os7212x0x7212ps28848x0x28848_output,
                d_minecraft_cave_layer_8_d5x1x5os19669x0x19669ps78678x0x78678_output,
                d_minecraft_realism_mountains_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_density_function_Multiply_10_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_realism_mountains_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_realism_mountains_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_caves_new_large with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_caves_new_large<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(1, 1, 1),
                make_double3(4, 16, 4),
                d_minecraft_determiner_cave_1_d5x132x5os65565x65565x65565ps262260x1049040x262260_output,
                d_minecraft_cave_layer_51_d5x132x5os163912x65565x163912ps655650x1049040x655650_output,
                d_minecraft_jagged_27_d5x132x5os16391250x0x16391250ps65565000x0x65565000_output,
                d_minecraft_caves_new_large_d5x132x5os65565x65565x65565ps262260x1049040x262260_output
            );
        }
        cudaStreamSynchronize(stream);

        // Wave 3: density_function_Multiply_18
        { // Kernel density_function_Multiply_18 with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            density_function_Multiply_18<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_gravel_17_d5x1x5os16391x0x16391ps65565x0x65565_output,
                d_density_function_Multiply_19_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_density_function_Multiply_18_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        cudaStreamSynchronize(stream);

        // Wave 4: minecraft_realism_extreme_hills_slope_height, minecraft_realism_hills_slope_height
        { // Kernel minecraft_realism_extreme_hills_slope_height with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_realism_extreme_hills_slope_height<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_realism_extreme_hills_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_density_function_Multiply_18_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_cave_layer_16_d5x1x5os13113x0x13113ps52452x0x52452_output,
                d_minecraft_realism_extreme_hills_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_realism_extreme_hills_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        { // Kernel minecraft_realism_hills_slope_height with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_realism_hills_slope_height<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_realism_hills_jagged_original_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_density_function_Multiply_18_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_cave_layer_25_d5x1x5os9834x0x9834ps39339x0x39339_output,
                d_minecraft_realism_hills_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_realism_extreme_hills_factor_original_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_realism_hills_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        cudaStreamSynchronize(stream);

        // Wave 5: minecraft_new_surface_height_unprocessed_unrivered
        { // Kernel minecraft_new_surface_height_unprocessed_unrivered with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_new_surface_height_unprocessed_unrivered<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_minecraft_realism_mountains_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_realism_extreme_hills_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_density_function_Abs_36_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_density_function_Add_29_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_density_function_Add_33_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_realism_extreme_mountains_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_realism_hills_slope_height_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_new_surface_height_unprocessed_unrivered_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        cudaStreamSynchronize(stream);

        // Wave 6: minecraft_new_surface_height_unprocessed_rivered
        { // Kernel minecraft_new_surface_height_unprocessed_rivered with dimensions 5x1x5
            int num_blocks = (25 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_new_surface_height_unprocessed_rivered<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 1, 5), origin,
                make_double3(1, 0, 1),
                make_double3(4, 0, 4),
                d_density_function_Abs_36_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_new_surface_height_unprocessed_unrivered_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_new_surface_height_unprocessed_rivered_d5x1x5os65565x0x65565ps262260x0x262260_output
            );
        }
        cudaStreamSynchronize(stream);

        // Wave 7: minecraft_caves_underlands_combined, minecraft_new_surface_combined_processed
        { // Kernel minecraft_caves_underlands_combined with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_caves_underlands_combined<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(1, 1, 1),
                make_double3(4, 16, 4),
                d_minecraft_new_surface_height_unprocessed_rivered_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_caves_underlands_combined_d5x132x5os65565x65565x65565ps262260x1049040x262260_output
            );
        }
        { // Kernel minecraft_new_surface_combined_processed with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_new_surface_combined_processed<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(1, 1, 1),
                make_double3(4, 16, 4),
                d_minecraft_new_surface_height_unprocessed_rivered_d5x1x5os65565x0x65565ps262260x0x262260_output,
                d_minecraft_new_surface_combined_processed_d5x132x5os65565x65565x65565ps262260x1049040x262260_output
            );
        }
        cudaStreamSynchronize(stream);

        // Wave 8: minecraft_new_combination_unfixed
        { // Kernel minecraft_new_combination_unfixed with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_new_combination_unfixed<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(1, 1, 1),
                make_double3(4, 16, 4),
                d_minecraft_caves_new_large_d5x132x5os65565x65565x65565ps262260x1049040x262260_output,
                d_minecraft_caves_new_medium_d5x132x5os65565x65565x65565ps262260x1049040x262260_output,
                d_minecraft_caves_old_medium_d5x132x5os65565x65565x65565ps262260x1049040x262260_output,
                d_minecraft_caves_barrier_2_d5x132x5os65565x65565x65565ps262260x1049040x262260_output,
                d_minecraft_new_surface_combined_processed_d5x132x5os65565x65565x65565ps262260x1049040x262260_output,
                d_minecraft_caves_old_small_d5x132x5os65565x65565x65565ps262260x1049040x262260_output,
                d_minecraft_caves_underlands_combined_d5x132x5os65565x65565x65565ps262260x1049040x262260_output,
                d_density_function_Noise_41_d5x132x5os65565x65565x65565ps262260x1049040x262260_output,
                d_minecraft_caves_barrier_1_d5x132x5os65565x65565x65565ps262260x1049040x262260_output,
                d_minecraft_caves_new_small_d5x132x5os65565x65565x65565ps262260x1049040x262260_output,
                d_minecraft_caves_old_large_d5x132x5os65565x65565x65565ps262260x1049040x262260_output,
                d_minecraft_new_combination_unfixed_d5x132x5os65565x65565x65565ps262260x1049040x262260_output
            );
        }
        cudaStreamSynchronize(stream);

        // Wave 9: minecraft_new_combination_fixed
        { // Kernel minecraft_new_combination_fixed with dimensions 5x132x5
            int num_blocks = (3300 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            minecraft_new_combination_fixed<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(5, 132, 5), origin,
                make_double3(1, 1, 1),
                make_double3(4, 16, 4),
                d_minecraft_new_combination_unfixed_d5x132x5os65565x65565x65565ps262260x1049040x262260_output,
                d_density_function_YClampedGradient_53_d5x132x5os65565x65565x65565ps262260x1049040x262260_output,
                d_minecraft_new_combination_fixed_d5x132x5os65565x65565x65565ps262260x1049040x262260_output
            );
        }
        cudaStreamSynchronize(stream);

        // Wave 10: final_density
        { // Kernel final_density with dimensions 16x2096x16
            int num_blocks = (536576 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            final_density<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                make_int3(0, 0, 0), make_int3(16, 2096, 16), origin,
                make_double3(1, 1, 1),
                make_double3(1, 1, 1),
                d_minecraft_new_combination_fixed_d5x132x5os65565x65565x65565ps262260x1049040x262260_output,
                d_final_density_d16x2096x16os65565x65565x65565ps65565x65565x65565_output
            );
        }
        cudaStreamSynchronize(stream);

        // Copy target output to host
        std::vector<double> result(536576);
        cudaMemcpyAsync(result.data(), d_final_density_d16x2096x16os65565x65565x65565ps65565x65565x65565_output, (size_t)536576 * sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return result;
    }

};

