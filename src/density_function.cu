// Auto-generated CUDA C++ module
#pragma once

#include "helpers.cu"

__constant__ __device__ const float _spline_coordinates_0_[41] = {-2.0f, -1.9f, -1.8f, -1.7f, -1.6f, -1.5f, -1.4f, -1.3f, -1.2f, -1.1f, -1.0f, -0.9f, -0.8f, -0.7f, -0.6f, -0.5f, -0.4f, -0.3f, -0.2f, -0.1f, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f};
__constant__ __device__ const float _spline_derivatives_1_[41] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
__constant__ __device__ const float _spline_coordinates_2_[81] = {-2.0f, -1.95f, -1.9f, -1.85f, -1.8f, -1.75f, -1.7f, -1.65f, -1.6f, -1.55f, -1.5f, -1.45f, -1.4f, -1.35f, -1.3f, -1.25f, -1.2f, -1.15f, -1.1f, -1.05f, -1.0f, -0.95f, -0.9f, -0.85f, -0.8f, -0.75f, -0.7f, -0.65f, -0.6f, -0.55f, -0.5f, -0.45f, -0.4f, -0.35f, -0.3f, -0.25f, -0.2f, -0.15f, -0.1f, -0.05f, 0.0f, 0.05f, 0.1f, 0.15f, 0.2f, 0.25f, 0.3f, 0.35f, 0.4f, 0.45f, 0.5f, 0.55f, 0.6f, 0.65f, 0.7f, 0.75f, 0.8f, 0.85f, 0.9f, 0.95f, 1.0f, 1.05f, 1.1f, 1.15f, 1.2f, 1.25f, 1.3f, 1.35f, 1.4f, 1.45f, 1.5f, 1.55f, 1.6f, 1.65f, 1.7f, 1.75f, 1.8f, 1.85f, 1.9f, 1.95f, 2.0f};
__constant__ __device__ const float _spline_derivatives_3_[81] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
__constant__ __device__ const float _spline_coordinates_4_[20] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
__constant__ __device__ const float _spline_derivatives_5_[20] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
__constant__ __device__ const float _spline_coordinates_6_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_7_[2] = {0.7f, 1.5f};
__constant__ __device__ const float _spline_coordinates_8_[2] = {0.0f, 1.0f};
__constant__ __device__ const float _spline_coordinates_9_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_10_[2] = {0.0f, 0.5f};
__constant__ __device__ const float _spline_coordinates_11_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_12_[2] = {0.0f, 0.5f};
__constant__ __device__ const float _spline_coordinates_13_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_14_[2] = {0.0f, 0.5f};
__constant__ __device__ const float _spline_coordinates_15_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_16_[4] = {0.1f, 0.35f, 0.7f, 1.0f};
__constant__ __device__ const float _spline_coordinates_17_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_18_[2] = {0.0f, 0.5f};
__constant__ __device__ const float _spline_coordinates_19_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_20_[2] = {0.0f, 0.5f};
__constant__ __device__ const float _spline_coordinates_21_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_22_[2] = {0.0f, 0.5f};
__constant__ __device__ const float _spline_coordinates_23_[4] = {0.1f, 0.35f, 0.7f, 1.0f};
__constant__ __device__ const float _spline_coordinates_24_[2] = {-1.0f, 1.0f};
__constant__ __device__ const float _spline_coordinates_25_[3] = {-0.01f, 0.02f, 0.05f};
__constant__ __device__ const float _spline_derivatives_26_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_27_[2] = {0.0f, 0.0f};
__constant__ __device__ const float _spline_derivatives_28_[2] = {0.0f, 0.0f};
__constant__ __device__ const float _spline_derivatives_29_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_30_[2] = {0.0f, 0.0f};
__constant__ __device__ const float _spline_derivatives_31_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_32_[2] = {0.0f, 0.0f};
__constant__ __device__ const float _spline_derivatives_33_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_34_[2] = {0.0f, 0.0f};
__constant__ __device__ const float _spline_derivatives_35_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_36_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
__constant__ __device__ const float _spline_derivatives_37_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_38_[2] = {0.0f, 0.0f};
__constant__ __device__ const float _spline_derivatives_39_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_40_[2] = {0.0f, 0.0f};
__constant__ __device__ const float _spline_derivatives_41_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_42_[2] = {0.0f, 0.0f};
__constant__ __device__ const float _spline_derivatives_43_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
__constant__ __device__ const float _spline_derivatives_44_[2] = {0.0f, 0.0f};
__constant__ __device__ const float _spline_derivatives_45_[3] = {0.0f, 0.0f, 0.0f};
__constant__ __device__ const float _spline_coordinates_46_[17] = {0.0f, 0.05f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f};
__constant__ __device__ const float _spline_coordinates_47_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_48_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_49_[2] = {-0.3f, 0.3f};
__constant__ __device__ const float _spline_derivatives_50_[17] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
__constant__ __device__ const float _spline_derivatives_51_[1] = {0.035f};
__constant__ __device__ const float _spline_derivatives_52_[1] = {0.045f};
__constant__ __device__ const float _spline_derivatives_53_[2] = {0.0f, 0.0f};
__constant__ __device__ const float _spline_coordinates_54_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_55_[2] = {-64.0f, -54.0f};
__constant__ __device__ const float _spline_derivatives_56_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_57_[2] = {0.0f, 0.0f};
__constant__ __device__ const float _spline_coordinates_58_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_59_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_60_[2] = {-0.2f, 0.2f};
__constant__ __device__ const float _spline_coordinates_61_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_62_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_63_[2] = {-0.3f, 0.3f};
__constant__ __device__ const float _spline_derivatives_64_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_65_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_66_[2] = {0.0f, 0.0f};
__constant__ __device__ const float _spline_derivatives_67_[1] = {0.03f};
__constant__ __device__ const float _spline_derivatives_68_[1] = {0.06f};
__constant__ __device__ const float _spline_derivatives_69_[2] = {0.0f, 0.0f};
__constant__ __device__ const float _spline_coordinates_70_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_71_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_72_[4] = {-0.35f, -0.2f, 0.2f, 0.35f};
__constant__ __device__ const float _spline_derivatives_73_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_74_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_75_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
__constant__ __device__ const float _spline_coordinates_76_[17] = {0.0f, 0.05f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f};
__constant__ __device__ const float _spline_derivatives_77_[17] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
__constant__ __device__ const float _spline_coordinates_78_[165] = {-2.0f, -1.975f, -1.95f, -1.925f, -1.9f, -1.875f, -1.85f, -1.825f, -1.8f, -1.775f, -1.75f, -1.725f, -1.7f, -1.675f, -1.65f, -1.625f, -1.6f, -1.575f, -1.55f, -1.525f, -1.5f, -1.475f, -1.45f, -1.425f, -1.4f, -1.375f, -1.35f, -1.325f, -1.3f, -1.275f, -1.25f, -1.225f, -1.2f, -1.175f, -1.15f, -1.125f, -1.1f, -1.075f, -1.05f, -1.025f, -1.1f, -1.075f, -1.05f, -1.025f, -1.0f, -0.975f, -0.95f, -0.925f, -0.9f, -0.875f, -0.85f, -0.825f, -0.8f, -0.775f, -0.75f, -0.725f, -0.7f, -0.675f, -0.65f, -0.625f, -0.6f, -0.575f, -0.55f, -0.525f, -0.5f, -0.475f, -0.45f, -0.425f, -0.4f, -0.375f, -0.35f, -0.325f, -0.3f, -0.275f, -0.25f, -0.225f, -0.2f, -0.175f, -0.15f, -0.125f, -0.1f, -0.075f, -0.05f, -0.025f, 0.0f, 0.025f, 0.05f, 0.075f, 0.1f, 0.125f, 0.15f, 0.175f, 0.2f, 0.225f, 0.25f, 0.275f, 0.3f, 0.325f, 0.35f, 0.375f, 0.4f, 0.425f, 0.45f, 0.475f, 0.5f, 0.525f, 0.55f, 0.575f, 0.6f, 0.625f, 0.65f, 0.675f, 0.7f, 0.725f, 0.75f, 0.775f, 0.8f, 0.825f, 0.85f, 0.875f, 0.9f, 0.925f, 0.95f, 0.975f, 1.0f, 1.025f, 1.05f, 1.075f, 1.1f, 1.125f, 1.15f, 1.175f, 1.2f, 1.225f, 1.25f, 1.275f, 1.3f, 1.325f, 1.35f, 1.375f, 1.4f, 1.425f, 1.45f, 1.475f, 1.5f, 1.525f, 1.55f, 1.575f, 1.6f, 1.625f, 1.65f, 1.675f, 1.7f, 1.725f, 1.75f, 1.775f, 1.8f, 1.825f, 1.85f, 1.875f, 1.9f, 1.925f, 1.95f, 1.975f, 2.0f};
__constant__ __device__ const float _spline_derivatives_79_[165] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
__constant__ __device__ const float _spline_coordinates_80_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_81_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_82_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_83_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_84_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_85_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_86_[6] = {0.0f, 0.075f, 0.15f, 0.225f, 0.3f, 0.4f};
__constant__ __device__ const float _spline_derivatives_87_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_88_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_89_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_90_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_91_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_92_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_93_[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
__constant__ __device__ const float _spline_coordinates_94_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_95_[3] = {0.3f, 0.32f, 0.34f};
__constant__ __device__ const float _spline_derivatives_96_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_97_[3] = {0.0f, 0.0f, 0.0f};
__constant__ __device__ const float _spline_coordinates_98_[17] = {0.0f, 0.05f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f};
__constant__ __device__ const float _spline_coordinates_99_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_100_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_101_[2] = {-0.3f, 0.3f};
__constant__ __device__ const float _spline_derivatives_102_[17] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
__constant__ __device__ const float _spline_derivatives_103_[1] = {0.04f};
__constant__ __device__ const float _spline_derivatives_104_[1] = {0.05f};
__constant__ __device__ const float _spline_derivatives_105_[2] = {0.0f, 0.0f};
__constant__ __device__ const float _spline_coordinates_106_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_107_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_108_[2] = {-0.2f, 0.2f};
__constant__ __device__ const float _spline_coordinates_109_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_110_[1] = {0.0f};
__constant__ __device__ const float _spline_coordinates_111_[2] = {-0.3f, 0.3f};
__constant__ __device__ const float _spline_derivatives_112_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_113_[1] = {1.0f};
__constant__ __device__ const float _spline_derivatives_114_[2] = {0.0f, 0.0f};
__constant__ __device__ const float _spline_derivatives_115_[1] = {0.03f};
__constant__ __device__ const float _spline_derivatives_116_[1] = {0.06f};
__constant__ __device__ const float _spline_derivatives_117_[2] = {0.0f, 0.0f};

__device__ double minecraft_gravel_shifted_12(double3 rpos3, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5) {
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f2;
    double3 rpos3f3;
    double n_8;
    double n_7;
    double n_6;
    double n_5;
    rpos3f0 = (rpos3 * 0.00390625);
    rpos3f1 = (rpos3 * 0.0078125);
    rpos3f2 = (rpos3 * 0.015625);
    rpos3f3 = (rpos3 * 0.03125);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_gravel_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__8)) * 0.5333333333333333);
    n_7 = ((perlin(rpos3f1, perm_table_minecraft_gravel_0_octave__7) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__7)) * 0.26666666666666666);
    n_6 = ((perlin(rpos3f2, perm_table_minecraft_gravel_0_octave__6) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__6)) * 0.13333333333333333);
    n_5 = ((perlin(rpos3f3, perm_table_minecraft_gravel_0_octave__5) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__5)) * 0.06666666666666667);
    return ((((n_8 + n_7) + n_6) + n_5) * 1.3333333333333333);
}

__device__ double minecraft_gravel_shifted_21(double3 rpos3, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5) {
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f2;
    double3 rpos3f3;
    double n_8;
    double n_7;
    double n_6;
    double n_5;
    rpos3f0 = (rpos3 * 0.00390625);
    rpos3f1 = (rpos3 * 0.0078125);
    rpos3f2 = (rpos3 * 0.015625);
    rpos3f3 = (rpos3 * 0.03125);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_gravel_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__8)) * 0.5333333333333333);
    n_7 = ((perlin(rpos3f1, perm_table_minecraft_gravel_0_octave__7) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__7)) * 0.26666666666666666);
    n_6 = ((perlin(rpos3f2, perm_table_minecraft_gravel_0_octave__6) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__6)) * 0.13333333333333333);
    n_5 = ((perlin(rpos3f3, perm_table_minecraft_gravel_0_octave__5) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__5)) * 0.06666666666666667);
    return ((((n_8 + n_7) + n_6) + n_5) * 1.3333333333333333);
}

__device__ double minecraft_gravel_shifted_4(double3 rpos3, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5) {
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f2;
    double3 rpos3f3;
    double n_8;
    double n_7;
    double n_6;
    double n_5;
    rpos3f0 = (rpos3 * 0.00390625);
    rpos3f1 = (rpos3 * 0.0078125);
    rpos3f2 = (rpos3 * 0.015625);
    rpos3f3 = (rpos3 * 0.03125);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_gravel_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__8)) * 0.5333333333333333);
    n_7 = ((perlin(rpos3f1, perm_table_minecraft_gravel_0_octave__7) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__7)) * 0.26666666666666666);
    n_6 = ((perlin(rpos3f2, perm_table_minecraft_gravel_0_octave__6) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__6)) * 0.13333333333333333);
    n_5 = ((perlin(rpos3f3, perm_table_minecraft_gravel_0_octave__5) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__5)) * 0.06666666666666667);
    return ((((n_8 + n_7) + n_6) + n_5) * 1.3333333333333333);
}

__device__ double minecraft_gravel_shifted_14(double3 rpos3, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5) {
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f2;
    double3 rpos3f3;
    double n_8;
    double n_7;
    double n_6;
    double n_5;
    rpos3f0 = (rpos3 * 0.00390625);
    rpos3f1 = (rpos3 * 0.0078125);
    rpos3f2 = (rpos3 * 0.015625);
    rpos3f3 = (rpos3 * 0.03125);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_gravel_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__8)) * 0.5333333333333333);
    n_7 = ((perlin(rpos3f1, perm_table_minecraft_gravel_0_octave__7) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__7)) * 0.26666666666666666);
    n_6 = ((perlin(rpos3f2, perm_table_minecraft_gravel_0_octave__6) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__6)) * 0.13333333333333333);
    n_5 = ((perlin(rpos3f3, perm_table_minecraft_gravel_0_octave__5) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__5)) * 0.06666666666666667);
    return ((((n_8 + n_7) + n_6) + n_5) * 1.3333333333333333);
}

__device__ double minecraft_gravel_shifted_23(double3 rpos3, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5) {
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f2;
    double3 rpos3f3;
    double n_8;
    double n_7;
    double n_6;
    double n_5;
    rpos3f0 = (rpos3 * 0.00390625);
    rpos3f1 = (rpos3 * 0.0078125);
    rpos3f2 = (rpos3 * 0.015625);
    rpos3f3 = (rpos3 * 0.03125);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_gravel_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__8)) * 0.5333333333333333);
    n_7 = ((perlin(rpos3f1, perm_table_minecraft_gravel_0_octave__7) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__7)) * 0.26666666666666666);
    n_6 = ((perlin(rpos3f2, perm_table_minecraft_gravel_0_octave__6) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__6)) * 0.13333333333333333);
    n_5 = ((perlin(rpos3f3, perm_table_minecraft_gravel_0_octave__5) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__5)) * 0.06666666666666667);
    return ((((n_8 + n_7) + n_6) + n_5) * 1.3333333333333333);
}

__device__ double minecraft_gravel_shifted_6(double3 rpos3, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5) {
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f2;
    double3 rpos3f3;
    double n_8;
    double n_7;
    double n_6;
    double n_5;
    rpos3f0 = (rpos3 * 0.00390625);
    rpos3f1 = (rpos3 * 0.0078125);
    rpos3f2 = (rpos3 * 0.015625);
    rpos3f3 = (rpos3 * 0.03125);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_gravel_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__8)) * 0.5333333333333333);
    n_7 = ((perlin(rpos3f1, perm_table_minecraft_gravel_0_octave__7) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__7)) * 0.26666666666666666);
    n_6 = ((perlin(rpos3f2, perm_table_minecraft_gravel_0_octave__6) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__6)) * 0.13333333333333333);
    n_5 = ((perlin(rpos3f3, perm_table_minecraft_gravel_0_octave__5) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__5)) * 0.06666666666666667);
    return ((((n_8 + n_7) + n_6) + n_5) * 1.3333333333333333);
}

__global__ void minecraft_realism_hills_jagged_original(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    output[tid] = result;
}

__global__ void minecraft_determiner_overworld_1(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    output[tid] = result;
}

__global__ void density_function_ShiftedNoise_11(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = minecraft_gravel_shifted_12(((rpos3 * make_double3(0.11, 0.0, 0.11)) + make_double3(1.0, 0.0, 0.0)), perm_table_minecraft_gravel_0_octave__8, perm_table_minecraft_gravel_1_octave__8, perm_table_minecraft_gravel_0_octave__7, perm_table_minecraft_gravel_1_octave__7, perm_table_minecraft_gravel_0_octave__6, perm_table_minecraft_gravel_1_octave__6, perm_table_minecraft_gravel_0_octave__5, perm_table_minecraft_gravel_1_octave__5);
    output[tid] = result;
}

__global__ void density_function_ShiftedNoise_20(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = minecraft_gravel_shifted_21(((rpos3 * make_double3(0.25, 0.0, 0.25)) + make_double3(1.0, 0.0, 0.0)), perm_table_minecraft_gravel_0_octave__8, perm_table_minecraft_gravel_1_octave__8, perm_table_minecraft_gravel_0_octave__7, perm_table_minecraft_gravel_1_octave__7, perm_table_minecraft_gravel_0_octave__6, perm_table_minecraft_gravel_1_octave__6, perm_table_minecraft_gravel_0_octave__5, perm_table_minecraft_gravel_1_octave__5);
    output[tid] = result;
}

__global__ void density_function_ShiftedNoise_3(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = minecraft_gravel_shifted_4(((rpos3 * make_double3(0.12, 0.0, 0.12)) + make_double3(1.0, 0.0, 0.0)), perm_table_minecraft_gravel_0_octave__8, perm_table_minecraft_gravel_1_octave__8, perm_table_minecraft_gravel_0_octave__7, perm_table_minecraft_gravel_1_octave__7, perm_table_minecraft_gravel_0_octave__6, perm_table_minecraft_gravel_1_octave__6, perm_table_minecraft_gravel_0_octave__5, perm_table_minecraft_gravel_1_octave__5);
    output[tid] = result;
}

__global__ void minecraft_caves_new_large(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, const double* input_1, const double* input_2, double* output) {
    float _coordinate_118_;
    int32_t _spline_index_119_;
    float _spline_result_f32_120_;
    double _spline_result_f64_121_;
    float _spline_values_122_[41];
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    {
        _spline_values_122_[0] = 1.0f;
        _spline_values_122_[1] = -1.0f;
        _spline_values_122_[2] = 1.0f;
        _spline_values_122_[3] = -1.0f;
        _spline_values_122_[4] = 1.0f;
        _spline_values_122_[5] = -1.0f;
        _spline_values_122_[6] = 1.0f;
        _spline_values_122_[7] = -1.0f;
        _spline_values_122_[8] = 1.0f;
        _spline_values_122_[9] = -1.0f;
        _spline_values_122_[10] = 1.0f;
        _spline_values_122_[11] = -1.0f;
        _spline_values_122_[12] = 1.0f;
        _spline_values_122_[13] = -1.0f;
        _spline_values_122_[14] = 1.0f;
        _spline_values_122_[15] = -1.0f;
        _spline_values_122_[16] = 1.0f;
        _spline_values_122_[17] = -1.0f;
        _spline_values_122_[18] = 1.0f;
        _spline_values_122_[19] = -1.0f;
        _spline_values_122_[20] = 1.0f;
        _spline_values_122_[21] = -1.0f;
        _spline_values_122_[22] = 1.0f;
        _spline_values_122_[23] = -1.0f;
        _spline_values_122_[24] = 1.0f;
        _spline_values_122_[25] = -1.0f;
        _spline_values_122_[26] = 1.0f;
        _spline_values_122_[27] = -1.0f;
        _spline_values_122_[28] = 1.0f;
        _spline_values_122_[29] = -1.0f;
        _spline_values_122_[30] = 1.0f;
        _spline_values_122_[31] = -1.0f;
        _spline_values_122_[32] = 1.0f;
        _spline_values_122_[33] = -1.0f;
        _spline_values_122_[34] = 1.0f;
        _spline_values_122_[35] = -1.0f;
        _spline_values_122_[36] = 1.0f;
        _spline_values_122_[37] = -1.0f;
        _spline_values_122_[38] = 1.0f;
        _spline_values_122_[39] = -1.0f;
        _spline_values_122_[40] = 1.0f;
    }
    _coordinate_118_ = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_119_ = binary_search(_spline_coordinates_0_, _coordinate_118_);
    _spline_result_f32_120_ = advanced_hermite(_spline_coordinates_0_, _spline_values_122_, _spline_derivatives_1_, _coordinate_118_, _spline_index_119_);
    _spline_result_f64_121_ = _spline_result_f32_120_;
    result = (_spline_result_f64_121_ + ((input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))] * 1.5) + (input_2[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))] * 0.5)));
    output[tid] = result;
}

__global__ void minecraft_realism_extreme_hills_jagged_original(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    output[tid] = result;
}

__global__ void minecraft_caves_new_medium(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, const double* input_1, const double* input_2, double* output) {
    float _coordinate_123_;
    int32_t _spline_index_124_;
    float _spline_result_f32_125_;
    double _spline_result_f64_126_;
    float _spline_values_127_[81];
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    {
        _spline_values_127_[0] = 1.0f;
        _spline_values_127_[1] = -1.0f;
        _spline_values_127_[2] = 1.0f;
        _spline_values_127_[3] = -1.0f;
        _spline_values_127_[4] = 1.0f;
        _spline_values_127_[5] = -1.0f;
        _spline_values_127_[6] = 1.0f;
        _spline_values_127_[7] = -1.0f;
        _spline_values_127_[8] = 1.0f;
        _spline_values_127_[9] = -1.0f;
        _spline_values_127_[10] = 1.0f;
        _spline_values_127_[11] = -1.0f;
        _spline_values_127_[12] = 1.0f;
        _spline_values_127_[13] = -1.0f;
        _spline_values_127_[14] = 1.0f;
        _spline_values_127_[15] = -1.0f;
        _spline_values_127_[16] = 1.0f;
        _spline_values_127_[17] = -1.0f;
        _spline_values_127_[18] = 1.0f;
        _spline_values_127_[19] = -1.0f;
        _spline_values_127_[20] = 1.0f;
        _spline_values_127_[21] = -1.0f;
        _spline_values_127_[22] = 1.0f;
        _spline_values_127_[23] = -1.0f;
        _spline_values_127_[24] = 1.0f;
        _spline_values_127_[25] = -1.0f;
        _spline_values_127_[26] = 1.0f;
        _spline_values_127_[27] = -1.0f;
        _spline_values_127_[28] = 1.0f;
        _spline_values_127_[29] = -1.0f;
        _spline_values_127_[30] = 1.0f;
        _spline_values_127_[31] = -1.0f;
        _spline_values_127_[32] = 1.0f;
        _spline_values_127_[33] = -1.0f;
        _spline_values_127_[34] = 1.0f;
        _spline_values_127_[35] = -1.0f;
        _spline_values_127_[36] = 1.0f;
        _spline_values_127_[37] = -1.0f;
        _spline_values_127_[38] = 1.0f;
        _spline_values_127_[39] = -1.0f;
        _spline_values_127_[40] = 1.0f;
        _spline_values_127_[41] = -1.0f;
        _spline_values_127_[42] = 1.0f;
        _spline_values_127_[43] = -1.0f;
        _spline_values_127_[44] = 1.0f;
        _spline_values_127_[45] = -1.0f;
        _spline_values_127_[46] = 1.0f;
        _spline_values_127_[47] = -1.0f;
        _spline_values_127_[48] = 1.0f;
        _spline_values_127_[49] = -1.0f;
        _spline_values_127_[50] = 1.0f;
        _spline_values_127_[51] = -1.0f;
        _spline_values_127_[52] = 1.0f;
        _spline_values_127_[53] = -1.0f;
        _spline_values_127_[54] = 1.0f;
        _spline_values_127_[55] = -1.0f;
        _spline_values_127_[56] = 1.0f;
        _spline_values_127_[57] = -1.0f;
        _spline_values_127_[58] = 1.0f;
        _spline_values_127_[59] = -1.0f;
        _spline_values_127_[60] = 1.0f;
        _spline_values_127_[61] = -1.0f;
        _spline_values_127_[62] = 1.0f;
        _spline_values_127_[63] = -1.0f;
        _spline_values_127_[64] = 1.0f;
        _spline_values_127_[65] = -1.0f;
        _spline_values_127_[66] = 1.0f;
        _spline_values_127_[67] = -1.0f;
        _spline_values_127_[68] = 1.0f;
        _spline_values_127_[69] = -1.0f;
        _spline_values_127_[70] = 1.0f;
        _spline_values_127_[71] = -1.0f;
        _spline_values_127_[72] = 1.0f;
        _spline_values_127_[73] = -1.0f;
        _spline_values_127_[74] = 1.0f;
        _spline_values_127_[75] = -1.0f;
        _spline_values_127_[76] = 1.0f;
        _spline_values_127_[77] = -1.0f;
        _spline_values_127_[78] = 1.0f;
        _spline_values_127_[79] = -1.0f;
        _spline_values_127_[80] = 1.0f;
    }
    _coordinate_123_ = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_124_ = binary_search(_spline_coordinates_2_, _coordinate_123_);
    _spline_result_f32_125_ = advanced_hermite(_spline_coordinates_2_, _spline_values_127_, _spline_derivatives_3_, _coordinate_123_, _spline_index_124_);
    _spline_result_f64_126_ = _spline_result_f32_125_;
    result = (_spline_result_f64_126_ + ((input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))] * 1.5) + (input_2[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))] * 0.5)));
    output[tid] = result;
}

__global__ void minecraft_gravel_34(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5, double* output) {
    double n_5;
    double n_6;
    double n_7;
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f2;
    double3 rpos3f3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(0.05, 0.0, 0.05)) + (pos3 * make_double3(0.2, 0.0, 0.2)));
    rpos3f0 = (rpos3 * 0.00390625);
    rpos3f1 = (rpos3 * 0.0078125);
    rpos3f2 = (rpos3 * 0.015625);
    rpos3f3 = (rpos3 * 0.03125);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_gravel_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__8)) * 0.5333333333333333);
    n_7 = ((perlin(rpos3f1, perm_table_minecraft_gravel_0_octave__7) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__7)) * 0.26666666666666666);
    n_6 = ((perlin(rpos3f2, perm_table_minecraft_gravel_0_octave__6) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__6)) * 0.13333333333333333);
    n_5 = ((perlin(rpos3f3, perm_table_minecraft_gravel_0_octave__5) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__5)) * 0.06666666666666667);
    result = ((((n_8 + n_7) + n_6) + n_5) * 1.3333333333333333);
    output[tid] = result;
}

__global__ void minecraft_gravel_39(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5, double* output) {
    double n_5;
    double n_6;
    double n_7;
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f2;
    double3 rpos3f3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(0.008, 0.0, 0.008)) + (pos3 * make_double3(0.032, 0.0, 0.032)));
    rpos3f0 = (rpos3 * 0.00390625);
    rpos3f1 = (rpos3 * 0.0078125);
    rpos3f2 = (rpos3 * 0.015625);
    rpos3f3 = (rpos3 * 0.03125);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_gravel_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__8)) * 0.5333333333333333);
    n_7 = ((perlin(rpos3f1, perm_table_minecraft_gravel_0_octave__7) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__7)) * 0.26666666666666666);
    n_6 = ((perlin(rpos3f2, perm_table_minecraft_gravel_0_octave__6) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__6)) * 0.13333333333333333);
    n_5 = ((perlin(rpos3f3, perm_table_minecraft_gravel_0_octave__5) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__5)) * 0.06666666666666667);
    result = ((((n_8 + n_7) + n_6) + n_5) * 1.3333333333333333);
    output[tid] = result;
}

__global__ void minecraft_gravel_31(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5, double* output) {
    double n_5;
    double n_6;
    double n_7;
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f2;
    double3 rpos3f3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(0.1, 0.0, 0.1)) + (pos3 * make_double3(0.4, 0.0, 0.4)));
    rpos3f0 = (rpos3 * 0.00390625);
    rpos3f1 = (rpos3 * 0.0078125);
    rpos3f2 = (rpos3 * 0.015625);
    rpos3f3 = (rpos3 * 0.03125);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_gravel_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__8)) * 0.5333333333333333);
    n_7 = ((perlin(rpos3f1, perm_table_minecraft_gravel_0_octave__7) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__7)) * 0.26666666666666666);
    n_6 = ((perlin(rpos3f2, perm_table_minecraft_gravel_0_octave__6) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__6)) * 0.13333333333333333);
    n_5 = ((perlin(rpos3f3, perm_table_minecraft_gravel_0_octave__5) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__5)) * 0.06666666666666667);
    result = ((((n_8 + n_7) + n_6) + n_5) * 1.3333333333333333);
    output[tid] = result;
}

__global__ void minecraft_gravel_28(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5, double* output) {
    double n_5;
    double n_6;
    double n_7;
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f2;
    double3 rpos3f3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(0.2, 0.0, 0.2)) + (pos3 * make_double3(0.8, 0.0, 0.8)));
    rpos3f0 = (rpos3 * 0.00390625);
    rpos3f1 = (rpos3 * 0.0078125);
    rpos3f2 = (rpos3 * 0.015625);
    rpos3f3 = (rpos3 * 0.03125);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_gravel_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__8)) * 0.5333333333333333);
    n_7 = ((perlin(rpos3f1, perm_table_minecraft_gravel_0_octave__7) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__7)) * 0.26666666666666666);
    n_6 = ((perlin(rpos3f2, perm_table_minecraft_gravel_0_octave__6) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__6)) * 0.13333333333333333);
    n_5 = ((perlin(rpos3f3, perm_table_minecraft_gravel_0_octave__5) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__5)) * 0.06666666666666667);
    result = ((((n_8 + n_7) + n_6) + n_5) * 1.3333333333333333);
    output[tid] = result;
}

__global__ void minecraft_gravel_1(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5, double* output) {
    double n_5;
    double n_6;
    double n_7;
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f2;
    double3 rpos3f3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(0.12, 0.0, 0.12)) + (pos3 * make_double3(0.48, 0.0, 0.48)));
    rpos3f0 = (rpos3 * 0.00390625);
    rpos3f1 = (rpos3 * 0.0078125);
    rpos3f2 = (rpos3 * 0.015625);
    rpos3f3 = (rpos3 * 0.03125);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_gravel_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__8)) * 0.5333333333333333);
    n_7 = ((perlin(rpos3f1, perm_table_minecraft_gravel_0_octave__7) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__7)) * 0.26666666666666666);
    n_6 = ((perlin(rpos3f2, perm_table_minecraft_gravel_0_octave__6) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__6)) * 0.13333333333333333);
    n_5 = ((perlin(rpos3f3, perm_table_minecraft_gravel_0_octave__5) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__5)) * 0.06666666666666667);
    result = ((((n_8 + n_7) + n_6) + n_5) * 1.3333333333333333);
    output[tid] = result;
}

__global__ void minecraft_gravel_17(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5, double* output) {
    double n_5;
    double n_6;
    double n_7;
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f2;
    double3 rpos3f3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(0.25, 0.0, 0.25)) + (pos3 * make_double3(1.0, 0.0, 1.0)));
    rpos3f0 = (rpos3 * 0.00390625);
    rpos3f1 = (rpos3 * 0.0078125);
    rpos3f2 = (rpos3 * 0.015625);
    rpos3f3 = (rpos3 * 0.03125);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_gravel_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__8)) * 0.5333333333333333);
    n_7 = ((perlin(rpos3f1, perm_table_minecraft_gravel_0_octave__7) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__7)) * 0.26666666666666666);
    n_6 = ((perlin(rpos3f2, perm_table_minecraft_gravel_0_octave__6) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__6)) * 0.13333333333333333);
    n_5 = ((perlin(rpos3f3, perm_table_minecraft_gravel_0_octave__5) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__5)) * 0.06666666666666667);
    result = ((((n_8 + n_7) + n_6) + n_5) * 1.3333333333333333);
    output[tid] = result;
}

__global__ void minecraft_gravel_47(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5, double* output) {
    double n_5;
    double n_6;
    double n_7;
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f2;
    double3 rpos3f3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(0.03, 0.0, 0.03)) + (pos3 * make_double3(0.12, 0.0, 0.12)));
    rpos3f0 = (rpos3 * 0.00390625);
    rpos3f1 = (rpos3 * 0.0078125);
    rpos3f2 = (rpos3 * 0.015625);
    rpos3f3 = (rpos3 * 0.03125);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_gravel_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__8)) * 0.5333333333333333);
    n_7 = ((perlin(rpos3f1, perm_table_minecraft_gravel_0_octave__7) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__7)) * 0.26666666666666666);
    n_6 = ((perlin(rpos3f2, perm_table_minecraft_gravel_0_octave__6) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__6)) * 0.13333333333333333);
    n_5 = ((perlin(rpos3f3, perm_table_minecraft_gravel_0_octave__5) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__5)) * 0.06666666666666667);
    result = ((((n_8 + n_7) + n_6) + n_5) * 1.3333333333333333);
    output[tid] = result;
}

__global__ void minecraft_gravel_9(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5, double* output) {
    double n_5;
    double n_6;
    double n_7;
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f2;
    double3 rpos3f3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(0.11, 0.0, 0.11)) + (pos3 * make_double3(0.44, 0.0, 0.44)));
    rpos3f0 = (rpos3 * 0.00390625);
    rpos3f1 = (rpos3 * 0.0078125);
    rpos3f2 = (rpos3 * 0.015625);
    rpos3f3 = (rpos3 * 0.03125);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_gravel_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__8)) * 0.5333333333333333);
    n_7 = ((perlin(rpos3f1, perm_table_minecraft_gravel_0_octave__7) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__7)) * 0.26666666666666666);
    n_6 = ((perlin(rpos3f2, perm_table_minecraft_gravel_0_octave__6) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__6)) * 0.13333333333333333);
    n_5 = ((perlin(rpos3f3, perm_table_minecraft_gravel_0_octave__5) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_gravel_1_octave__5)) * 0.06666666666666667);
    result = ((((n_8 + n_7) + n_6) + n_5) * 1.3333333333333333);
    output[tid] = result;
}

__global__ void density_function_ShiftedNoise_13(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = minecraft_gravel_shifted_14(((rpos3 * make_double3(0.11, 0.0, 0.11)) + make_double3(0.0, 0.0, 1.0)), perm_table_minecraft_gravel_0_octave__8, perm_table_minecraft_gravel_1_octave__8, perm_table_minecraft_gravel_0_octave__7, perm_table_minecraft_gravel_1_octave__7, perm_table_minecraft_gravel_0_octave__6, perm_table_minecraft_gravel_1_octave__6, perm_table_minecraft_gravel_0_octave__5, perm_table_minecraft_gravel_1_octave__5);
    output[tid] = result;
}

__global__ void density_function_ShiftedNoise_22(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = minecraft_gravel_shifted_23(((rpos3 * make_double3(0.25, 0.0, 0.25)) + make_double3(0.0, 0.0, 1.0)), perm_table_minecraft_gravel_0_octave__8, perm_table_minecraft_gravel_1_octave__8, perm_table_minecraft_gravel_0_octave__7, perm_table_minecraft_gravel_1_octave__7, perm_table_minecraft_gravel_0_octave__6, perm_table_minecraft_gravel_1_octave__6, perm_table_minecraft_gravel_0_octave__5, perm_table_minecraft_gravel_1_octave__5);
    output[tid] = result;
}

__global__ void density_function_ShiftedNoise_5(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_gravel_0_octave__8, const int8_t* perm_table_minecraft_gravel_1_octave__8, const int8_t* perm_table_minecraft_gravel_0_octave__7, const int8_t* perm_table_minecraft_gravel_1_octave__7, const int8_t* perm_table_minecraft_gravel_0_octave__6, const int8_t* perm_table_minecraft_gravel_1_octave__6, const int8_t* perm_table_minecraft_gravel_0_octave__5, const int8_t* perm_table_minecraft_gravel_1_octave__5, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = minecraft_gravel_shifted_6(((rpos3 * make_double3(0.12, 0.0, 0.12)) + make_double3(0.0, 0.0, 1.0)), perm_table_minecraft_gravel_0_octave__8, perm_table_minecraft_gravel_1_octave__8, perm_table_minecraft_gravel_0_octave__7, perm_table_minecraft_gravel_1_octave__7, perm_table_minecraft_gravel_0_octave__6, perm_table_minecraft_gravel_1_octave__6, perm_table_minecraft_gravel_0_octave__5, perm_table_minecraft_gravel_1_octave__5);
    output[tid] = result;
}

__global__ void density_function_Multiply_19(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    float _coordinate_128_;
    int32_t _spline_index_129_;
    float _spline_result_f32_130_;
    double _spline_result_f64_131_;
    float _spline_values_132_[20];
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    {
        _spline_values_132_[0] = 0.0f;
        _spline_values_132_[1] = 0.31622776f;
        _spline_values_132_[2] = 0.4472136f;
        _spline_values_132_[3] = 0.5477226f;
        _spline_values_132_[4] = 0.6324555f;
        _spline_values_132_[5] = 0.70710677f;
        _spline_values_132_[6] = 0.7745967f;
        _spline_values_132_[7] = 0.83666f;
        _spline_values_132_[8] = 0.8944272f;
        _spline_values_132_[9] = 0.9486833f;
        _spline_values_132_[10] = 1.0f;
        _spline_values_132_[11] = 1.4142135f;
        _spline_values_132_[12] = 1.7320508f;
        _spline_values_132_[13] = 2.0f;
        _spline_values_132_[14] = 2.236068f;
        _spline_values_132_[15] = 2.4494898f;
        _spline_values_132_[16] = 2.6457512f;
        _spline_values_132_[17] = 2.828427f;
        _spline_values_132_[18] = 3.0f;
        _spline_values_132_[19] = 3.1622777f;
    }
    _coordinate_128_ = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_129_ = binary_search(_spline_coordinates_4_, _coordinate_128_);
    _spline_result_f32_130_ = advanced_hermite(_spline_coordinates_4_, _spline_values_132_, _spline_derivatives_5_, _coordinate_128_, _spline_index_129_);
    _spline_result_f64_131_ = _spline_result_f32_130_;
    result = (_spline_result_f64_131_ * 0.8);
    output[tid] = result;
}

__global__ void minecraft_caves_old_medium(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    output[tid] = result;
}

__global__ void minecraft_caves_barrier_2(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    output[tid] = result;
}

__global__ void minecraft_new_surface_combined_processed(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    double _var_133_;
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    _var_133_ = clamp(((rpos3.y - -64.0) / 2096.0), 0.0, 1.0);
    result = (((_var_133_ * -2.0) + 1.0) + input_0[flat_y_zero_index(pos3, 5, 5)]);
    output[tid] = result;
}

__global__ void minecraft_caves_old_small(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    output[tid] = result;
}

__global__ void minecraft_new_combination_unfixed(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, const double* input_1, const double* input_2, const double* input_3, const double* input_4, const double* input_5, const double* input_6, const double* input_7, const double* input_8, const double* input_9, const double* input_10, double* output) {
    float _coordinate_134_;
    float _coordinate_135_;
    float _coordinate_136_;
    float _coordinate_137_;
    float _coordinate_138_;
    float _coordinate_139_;
    float _coordinate_140_;
    float _coordinate_141_;
    float _coordinate_142_;
    float _coordinate_143_;
    float _coordinate_144_;
    float _coordinate_145_;
    float _coordinate_146_;
    float _coordinate_147_;
    float _coordinate_148_;
    float _coordinate_149_;
    float _coordinate_150_;
    float _coordinate_151_;
    float _coordinate_152_;
    float _coordinate_153_;
    int32_t _spline_index_154_;
    int32_t _spline_index_155_;
    int32_t _spline_index_156_;
    int32_t _spline_index_157_;
    int32_t _spline_index_158_;
    int32_t _spline_index_159_;
    int32_t _spline_index_160_;
    int32_t _spline_index_161_;
    int32_t _spline_index_162_;
    int32_t _spline_index_163_;
    int32_t _spline_index_164_;
    int32_t _spline_index_165_;
    int32_t _spline_index_166_;
    int32_t _spline_index_167_;
    int32_t _spline_index_168_;
    int32_t _spline_index_169_;
    int32_t _spline_index_170_;
    int32_t _spline_index_171_;
    int32_t _spline_index_172_;
    int32_t _spline_index_173_;
    float _spline_result_f32_174_;
    double _spline_result_f64_175_;
    float _spline_values_176_[1];
    float _spline_values_177_[2];
    float _spline_values_178_[2];
    float _spline_values_179_[1];
    float _spline_values_180_[2];
    float _spline_values_181_[1];
    float _spline_values_182_[2];
    float _spline_values_183_[1];
    float _spline_values_184_[2];
    float _spline_values_185_[1];
    float _spline_values_186_[4];
    float _spline_values_187_[1];
    float _spline_values_188_[2];
    float _spline_values_189_[1];
    float _spline_values_190_[2];
    float _spline_values_191_[1];
    float _spline_values_192_[2];
    float _spline_values_193_[4];
    float _spline_values_194_[2];
    float _spline_values_195_[3];
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    {
        _spline_values_176_[0] = 0.0f;
    }
    _coordinate_134_ = input_4[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_154_ = binary_search(_spline_coordinates_6_, _coordinate_134_);
    {
        _spline_values_177_[0] = advanced_hermite(_spline_coordinates_6_, _spline_values_176_, _spline_derivatives_26_, _coordinate_134_, _spline_index_154_);
        _spline_values_177_[1] = -10.0f;
    }
    _coordinate_135_ = input_7[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_155_ = binary_search(_spline_coordinates_7_, _coordinate_135_);
    {
        _spline_values_178_[0] = advanced_hermite(_spline_coordinates_6_, _spline_values_176_, _spline_derivatives_26_, _coordinate_134_, _spline_index_154_);
        _spline_values_178_[1] = advanced_hermite(_spline_coordinates_7_, _spline_values_177_, _spline_derivatives_27_, _coordinate_135_, _spline_index_155_);
    }
    _coordinate_136_ = input_3[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_156_ = binary_search(_spline_coordinates_8_, _coordinate_136_);
    {
        _spline_values_179_[0] = 0.0f;
    }
    _coordinate_137_ = input_5[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_157_ = binary_search(_spline_coordinates_9_, _coordinate_137_);
    {
        _spline_values_180_[0] = 0.1f;
        _spline_values_180_[1] = advanced_hermite(_spline_coordinates_9_, _spline_values_179_, _spline_derivatives_29_, _coordinate_137_, _spline_index_157_);
    }
    _coordinate_138_ = input_3[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_158_ = binary_search(_spline_coordinates_10_, _coordinate_138_);
    {
        _spline_values_181_[0] = 0.0f;
    }
    _coordinate_139_ = input_2[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_159_ = binary_search(_spline_coordinates_11_, _coordinate_139_);
    {
        _spline_values_182_[0] = 0.1f;
        _spline_values_182_[1] = advanced_hermite(_spline_coordinates_11_, _spline_values_181_, _spline_derivatives_31_, _coordinate_139_, _spline_index_159_);
    }
    _coordinate_140_ = input_3[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_160_ = binary_search(_spline_coordinates_12_, _coordinate_140_);
    {
        _spline_values_183_[0] = 0.0f;
    }
    _coordinate_141_ = input_10[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_161_ = binary_search(_spline_coordinates_13_, _coordinate_141_);
    {
        _spline_values_184_[0] = 0.1f;
        _spline_values_184_[1] = advanced_hermite(_spline_coordinates_13_, _spline_values_183_, _spline_derivatives_33_, _coordinate_141_, _spline_index_161_);
    }
    _coordinate_142_ = input_3[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_162_ = binary_search(_spline_coordinates_14_, _coordinate_142_);
    {
        _spline_values_185_[0] = 0.0f;
    }
    _coordinate_143_ = input_6[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_163_ = binary_search(_spline_coordinates_15_, _coordinate_143_);
    {
        _spline_values_186_[0] = advanced_hermite(_spline_coordinates_10_, _spline_values_180_, _spline_derivatives_30_, _coordinate_138_, _spline_index_158_);
        _spline_values_186_[1] = advanced_hermite(_spline_coordinates_12_, _spline_values_182_, _spline_derivatives_32_, _coordinate_140_, _spline_index_160_);
        _spline_values_186_[2] = advanced_hermite(_spline_coordinates_14_, _spline_values_184_, _spline_derivatives_34_, _coordinate_142_, _spline_index_162_);
        _spline_values_186_[3] = advanced_hermite(_spline_coordinates_15_, _spline_values_185_, _spline_derivatives_35_, _coordinate_143_, _spline_index_163_);
    }
    _coordinate_144_ = input_4[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_164_ = binary_search(_spline_coordinates_16_, _coordinate_144_);
    {
        _spline_values_187_[0] = 0.0f;
    }
    _coordinate_145_ = input_9[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_165_ = binary_search(_spline_coordinates_17_, _coordinate_145_);
    {
        _spline_values_188_[0] = 0.1f;
        _spline_values_188_[1] = advanced_hermite(_spline_coordinates_17_, _spline_values_187_, _spline_derivatives_37_, _coordinate_145_, _spline_index_165_);
    }
    _coordinate_146_ = input_3[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_166_ = binary_search(_spline_coordinates_18_, _coordinate_146_);
    {
        _spline_values_189_[0] = 0.0f;
    }
    _coordinate_147_ = input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_167_ = binary_search(_spline_coordinates_19_, _coordinate_147_);
    {
        _spline_values_190_[0] = 0.1f;
        _spline_values_190_[1] = advanced_hermite(_spline_coordinates_19_, _spline_values_189_, _spline_derivatives_39_, _coordinate_147_, _spline_index_167_);
    }
    _coordinate_148_ = input_3[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_168_ = binary_search(_spline_coordinates_20_, _coordinate_148_);
    {
        _spline_values_191_[0] = 0.0f;
    }
    _coordinate_149_ = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_169_ = binary_search(_spline_coordinates_21_, _coordinate_149_);
    {
        _spline_values_192_[0] = 0.1f;
        _spline_values_192_[1] = advanced_hermite(_spline_coordinates_21_, _spline_values_191_, _spline_derivatives_41_, _coordinate_149_, _spline_index_169_);
    }
    _coordinate_150_ = input_3[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_170_ = binary_search(_spline_coordinates_22_, _coordinate_150_);
    {
        _spline_values_193_[0] = advanced_hermite(_spline_coordinates_18_, _spline_values_188_, _spline_derivatives_38_, _coordinate_146_, _spline_index_166_);
        _spline_values_193_[1] = advanced_hermite(_spline_coordinates_20_, _spline_values_190_, _spline_derivatives_40_, _coordinate_148_, _spline_index_168_);
        _spline_values_193_[2] = advanced_hermite(_spline_coordinates_22_, _spline_values_192_, _spline_derivatives_42_, _coordinate_150_, _spline_index_170_);
        _spline_values_193_[3] = advanced_hermite(_spline_coordinates_15_, _spline_values_185_, _spline_derivatives_35_, _coordinate_143_, _spline_index_163_);
    }
    _coordinate_151_ = input_4[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_171_ = binary_search(_spline_coordinates_23_, _coordinate_151_);
    {
        _spline_values_194_[0] = advanced_hermite(_spline_coordinates_16_, _spline_values_186_, _spline_derivatives_36_, _coordinate_144_, _spline_index_164_);
        _spline_values_194_[1] = advanced_hermite(_spline_coordinates_23_, _spline_values_193_, _spline_derivatives_43_, _coordinate_151_, _spline_index_171_);
    }
    _coordinate_152_ = input_8[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_172_ = binary_search(_spline_coordinates_24_, _coordinate_152_);
    {
        _spline_values_195_[0] = advanced_hermite(_spline_coordinates_6_, _spline_values_176_, _spline_derivatives_26_, _coordinate_134_, _spline_index_154_);
        _spline_values_195_[1] = advanced_hermite(_spline_coordinates_8_, _spline_values_178_, _spline_derivatives_28_, _coordinate_136_, _spline_index_156_);
        _spline_values_195_[2] = advanced_hermite(_spline_coordinates_24_, _spline_values_194_, _spline_derivatives_44_, _coordinate_152_, _spline_index_172_);
    }
    _coordinate_153_ = input_4[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_173_ = binary_search(_spline_coordinates_25_, _coordinate_153_);
    _spline_result_f32_174_ = advanced_hermite(_spline_coordinates_25_, _spline_values_195_, _spline_derivatives_45_, _coordinate_153_, _spline_index_173_);
    _spline_result_f64_175_ = _spline_result_f32_174_;
    result = _spline_result_f64_175_;
    output[tid] = result;
}

__global__ void minecraft_realism_mountains_slope_height(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, const double* input_1, const double* input_2, const double* input_3, const double* input_4, double* output) {
    float _coordinate_196_;
    float _coordinate_197_;
    float _coordinate_198_;
    float _coordinate_199_;
    int32_t _spline_index_200_;
    int32_t _spline_index_201_;
    int32_t _spline_index_202_;
    int32_t _spline_index_203_;
    float _spline_result_f32_204_;
    float _spline_result_f32_205_;
    double _spline_result_f64_206_;
    double _spline_result_f64_207_;
    float _spline_values_208_[17];
    float _spline_values_209_[1];
    float _spline_values_210_[1];
    float _spline_values_211_[2];
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    {
        _spline_values_208_[0] = 1.0f;
        _spline_values_208_[1] = 0.95238096f;
        _spline_values_208_[2] = 0.90909094f;
        _spline_values_208_[3] = 0.8333333f;
        _spline_values_208_[4] = 0.7692308f;
        _spline_values_208_[5] = 0.71428573f;
        _spline_values_208_[6] = 0.6666667f;
        _spline_values_208_[7] = 0.625f;
        _spline_values_208_[8] = 0.5882353f;
        _spline_values_208_[9] = 0.5555556f;
        _spline_values_208_[10] = 0.5263158f;
        _spline_values_208_[11] = 0.5f;
        _spline_values_208_[12] = 0.47619048f;
        _spline_values_208_[13] = 0.45454547f;
        _spline_values_208_[14] = 0.4347826f;
        _spline_values_208_[15] = 0.41666666f;
        _spline_values_208_[16] = 0.4f;
    }
    _coordinate_196_ = input_3[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_200_ = binary_search(_spline_coordinates_46_, _coordinate_196_);
    _spline_result_f32_204_ = advanced_hermite(_spline_coordinates_46_, _spline_values_208_, _spline_derivatives_50_, _coordinate_196_, _spline_index_200_);
    _spline_result_f64_206_ = _spline_result_f32_204_;
    {
        _spline_values_209_[0] = 0.0f;
    }
    _coordinate_197_ = input_2[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_201_ = binary_search(_spline_coordinates_47_, _coordinate_197_);
    {
        _spline_values_210_[0] = 0.0f;
    }
    _coordinate_198_ = input_2[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_202_ = binary_search(_spline_coordinates_48_, _coordinate_198_);
    {
        _spline_values_211_[0] = advanced_hermite(_spline_coordinates_47_, _spline_values_209_, _spline_derivatives_51_, _coordinate_197_, _spline_index_201_);
        _spline_values_211_[1] = advanced_hermite(_spline_coordinates_48_, _spline_values_210_, _spline_derivatives_52_, _coordinate_198_, _spline_index_202_);
    }
    _coordinate_199_ = input_4[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_203_ = binary_search(_spline_coordinates_49_, _coordinate_199_);
    _spline_result_f32_205_ = advanced_hermite(_spline_coordinates_49_, _spline_values_211_, _spline_derivatives_53_, _coordinate_199_, _spline_index_203_);
    _spline_result_f64_207_ = _spline_result_f32_205_;
    result = ((0.0 + ((input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * 0.2) + 0.0)) + (((input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * _spline_result_f64_206_) * 0.7) + _spline_result_f64_207_));
    output[tid] = result;
}

__global__ void density_function_Multiply_2(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, const double* input_1, const double* input_2, double* output) {
    double _var_212_;
    double _var_213_;
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    _var_212_ = (input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] + (input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * -1.0));
    _var_213_ = (input_2[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] + (input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * -1.0));
    result = ((((_var_212_ * _var_212_) + (_var_213_ * _var_213_)) * 5000.0) * 0.2);
    output[tid] = result;
}

__global__ void minecraft_realism_extreme_mountains_jagged_original(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    output[tid] = result;
}

__global__ void minecraft_determiner_cave_1(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    double _var_214_;
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    _var_214_ = clamp(((rpos3.y - -64.0) / 2096.0), 0.0, 1.0);
    result = (((_var_214_ * -2.0) + 1.0) + input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))]);
    output[tid] = result;
}

__global__ void minecraft_realism_extreme_mountains_factor_original(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    output[tid] = result;
}

__global__ void minecraft_new_combination_fixed(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, const double* input_1, double* output) {
    float _coordinate_215_;
    float _coordinate_216_;
    int32_t _spline_index_217_;
    int32_t _spline_index_218_;
    float _spline_result_f32_219_;
    double _spline_result_f64_220_;
    float _spline_values_221_[1];
    float _spline_values_222_[2];
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    {
        _spline_values_221_[0] = 0.0f;
    }
    _coordinate_215_ = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_217_ = binary_search(_spline_coordinates_54_, _coordinate_215_);
    {
        _spline_values_222_[0] = 0.1f;
        _spline_values_222_[1] = advanced_hermite(_spline_coordinates_54_, _spline_values_221_, _spline_derivatives_56_, _coordinate_215_, _spline_index_217_);
    }
    _coordinate_216_ = input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_218_ = binary_search(_spline_coordinates_55_, _coordinate_216_);
    _spline_result_f32_219_ = advanced_hermite(_spline_coordinates_55_, _spline_values_222_, _spline_derivatives_57_, _coordinate_216_, _spline_index_218_);
    _spline_result_f64_220_ = _spline_result_f32_219_;
    result = _spline_result_f64_220_;
    output[tid] = result;
}

__global__ void density_function_YClampedGradient_53(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, double* output) {
    double _var_223_;
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    _var_223_ = clamp(((rpos3.y - -64.0) / 2096.0), 0.0, 1.0);
    result = ((_var_223_ * 2096.0) + -64.0);
    output[tid] = result;
}

__global__ void minecraft_realism_extreme_hills_slope_height(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, const double* input_1, const double* input_2, const double* input_3, double* output) {
    float _coordinate_224_;
    float _coordinate_225_;
    float _coordinate_226_;
    float _coordinate_227_;
    float _coordinate_228_;
    float _coordinate_229_;
    int32_t _spline_index_230_;
    int32_t _spline_index_231_;
    int32_t _spline_index_232_;
    int32_t _spline_index_233_;
    int32_t _spline_index_234_;
    int32_t _spline_index_235_;
    float _spline_result_f32_236_;
    float _spline_result_f32_237_;
    double _spline_result_f64_238_;
    double _spline_result_f64_239_;
    float _spline_values_240_[1];
    float _spline_values_241_[1];
    float _spline_values_242_[2];
    float _spline_values_243_[1];
    float _spline_values_244_[1];
    float _spline_values_245_[2];
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    {
        _spline_values_240_[0] = 0.0f;
    }
    _coordinate_224_ = input_3[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_230_ = binary_search(_spline_coordinates_58_, _coordinate_224_);
    {
        _spline_values_241_[0] = 0.0f;
    }
    _coordinate_225_ = input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_231_ = binary_search(_spline_coordinates_59_, _coordinate_225_);
    {
        _spline_values_242_[0] = advanced_hermite(_spline_coordinates_58_, _spline_values_240_, _spline_derivatives_64_, _coordinate_224_, _spline_index_230_);
        _spline_values_242_[1] = advanced_hermite(_spline_coordinates_59_, _spline_values_241_, _spline_derivatives_65_, _coordinate_225_, _spline_index_231_);
    }
    _coordinate_226_ = input_3[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_232_ = binary_search(_spline_coordinates_60_, _coordinate_226_);
    _spline_result_f32_236_ = advanced_hermite(_spline_coordinates_60_, _spline_values_242_, _spline_derivatives_66_, _coordinate_226_, _spline_index_232_);
    _spline_result_f64_238_ = _spline_result_f32_236_;
    {
        _spline_values_243_[0] = 0.0f;
    }
    _coordinate_227_ = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_233_ = binary_search(_spline_coordinates_61_, _coordinate_227_);
    {
        _spline_values_244_[0] = 0.0f;
    }
    _coordinate_228_ = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_234_ = binary_search(_spline_coordinates_62_, _coordinate_228_);
    {
        _spline_values_245_[0] = advanced_hermite(_spline_coordinates_61_, _spline_values_243_, _spline_derivatives_67_, _coordinate_227_, _spline_index_233_);
        _spline_values_245_[1] = advanced_hermite(_spline_coordinates_62_, _spline_values_244_, _spline_derivatives_68_, _coordinate_228_, _spline_index_234_);
    }
    _coordinate_229_ = input_3[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_235_ = binary_search(_spline_coordinates_63_, _coordinate_229_);
    _spline_result_f32_237_ = advanced_hermite(_spline_coordinates_63_, _spline_values_245_, _spline_derivatives_69_, _coordinate_229_, _spline_index_235_);
    _spline_result_f64_239_ = _spline_result_f32_237_;
    result = ((0.0 + ((input_2[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * 0.4) + -0.2)) + ((_spline_result_f64_238_ * 0.3) + _spline_result_f64_239_));
    output[tid] = result;
}

__global__ void minecraft_caves_underlands_combined(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    double _var_246_;
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    _var_246_ = clamp(((rpos3.y - -256.0) / 1272.0), 0.0, 1.0);
    result = (((_var_246_ * -2.0) + 1.0) + (input_0[flat_y_zero_index(pos3, 5, 5)] * -1.6));
    output[tid] = result;
}

__global__ void final_density(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = interpolate(input_0[cornerx0y0z0_16(pos3, 5, 132, 5)], input_0[cornerx4y0z0_16(pos3, 5, 132, 5)], input_0[cornerx0y16z0_16(pos3, 5, 132, 5)], input_0[cornerx4y16z0_16(pos3, 5, 132, 5)], input_0[cornerx0y0z4_16(pos3, 5, 132, 5)], input_0[cornerx4y0z4_16(pos3, 5, 132, 5)], input_0[cornerx0y16z4_16(pos3, 5, 132, 5)], input_0[cornerx4y16z4_16(pos3, 5, 132, 5)], xfract4(pos3), yfract16(pos3), zfract4(pos3));
    output[tid] = result;
}

__global__ void density_function_Noise_41(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    output[tid] = result;
}

__global__ void density_function_Abs_36(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, const double* input_1, double* output) {
    float _coordinate_247_;
    float _coordinate_248_;
    float _coordinate_249_;
    int32_t _spline_index_250_;
    int32_t _spline_index_251_;
    int32_t _spline_index_252_;
    float _spline_result_f32_253_;
    double _spline_result_f64_254_;
    float _spline_values_255_[1];
    float _spline_values_256_[1];
    float _spline_values_257_[4];
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    {
        _spline_values_255_[0] = 0.0f;
    }
    _coordinate_247_ = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_250_ = binary_search(_spline_coordinates_70_, _coordinate_247_);
    {
        _spline_values_256_[0] = 0.0f;
    }
    _coordinate_248_ = input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_251_ = binary_search(_spline_coordinates_71_, _coordinate_248_);
    {
        _spline_values_257_[0] = advanced_hermite(_spline_coordinates_70_, _spline_values_255_, _spline_derivatives_73_, _coordinate_247_, _spline_index_250_);
        _spline_values_257_[1] = advanced_hermite(_spline_coordinates_71_, _spline_values_256_, _spline_derivatives_74_, _coordinate_248_, _spline_index_251_);
        _spline_values_257_[2] = advanced_hermite(_spline_coordinates_71_, _spline_values_256_, _spline_derivatives_74_, _coordinate_248_, _spline_index_251_);
        _spline_values_257_[3] = advanced_hermite(_spline_coordinates_70_, _spline_values_255_, _spline_derivatives_73_, _coordinate_247_, _spline_index_250_);
    }
    _coordinate_249_ = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_252_ = binary_search(_spline_coordinates_72_, _coordinate_249_);
    _spline_result_f32_253_ = advanced_hermite(_spline_coordinates_72_, _spline_values_257_, _spline_derivatives_75_, _coordinate_249_, _spline_index_252_);
    _spline_result_f64_254_ = _spline_result_f32_253_;
    result = abs(_spline_result_f64_254_);
    output[tid] = result;
}

__global__ void minecraft_caves_barrier_1(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    output[tid] = result;
}

__global__ void density_function_Multiply_18(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, const double* input_1, double* output) {
    float _coordinate_258_;
    int32_t _spline_index_259_;
    float _spline_result_f32_260_;
    double _spline_result_f64_261_;
    float _spline_values_262_[17];
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    {
        _spline_values_262_[0] = 1.0f;
        _spline_values_262_[1] = 0.95238096f;
        _spline_values_262_[2] = 0.90909094f;
        _spline_values_262_[3] = 0.8333333f;
        _spline_values_262_[4] = 0.7692308f;
        _spline_values_262_[5] = 0.71428573f;
        _spline_values_262_[6] = 0.6666667f;
        _spline_values_262_[7] = 0.625f;
        _spline_values_262_[8] = 0.5882353f;
        _spline_values_262_[9] = 0.5555556f;
        _spline_values_262_[10] = 0.5263158f;
        _spline_values_262_[11] = 0.5f;
        _spline_values_262_[12] = 0.47619048f;
        _spline_values_262_[13] = 0.45454547f;
        _spline_values_262_[14] = 0.4347826f;
        _spline_values_262_[15] = 0.41666666f;
        _spline_values_262_[16] = 0.4f;
    }
    _coordinate_258_ = input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_259_ = binary_search(_spline_coordinates_76_, _coordinate_258_);
    _spline_result_f32_260_ = advanced_hermite(_spline_coordinates_76_, _spline_values_262_, _spline_derivatives_77_, _coordinate_258_, _spline_index_259_);
    _spline_result_f64_261_ = _spline_result_f32_260_;
    result = (input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * (_spline_result_f64_261_ * 0.3));
    output[tid] = result;
}

__global__ void minecraft_cave_layer_43(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_cave_layer_0_octave__8, const int8_t* perm_table_minecraft_cave_layer_1_octave__8, double* output) {
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = (origin + (pos3 * make_double3(4.0, 16.0, 4.0)));
    rpos3f0 = (rpos3 * 0.00390625);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_cave_layer_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_cave_layer_1_octave__8)) * 1.0);
    result = (n_8 * 0.8333333333333333);
    output[tid] = result;
}

__global__ void minecraft_cave_layer_42(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_cave_layer_0_octave__8, const int8_t* perm_table_minecraft_cave_layer_1_octave__8, double* output) {
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(2.0, 0.0, 2.0)) + (pos3 * make_double3(8.0, 0.0, 8.0)));
    rpos3f0 = (rpos3 * 0.00390625);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_cave_layer_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_cave_layer_1_octave__8)) * 1.0);
    result = (n_8 * 0.8333333333333333);
    output[tid] = result;
}

__global__ void minecraft_cave_layer_48(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_cave_layer_0_octave__8, const int8_t* perm_table_minecraft_cave_layer_1_octave__8, double* output) {
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(10.0, 4.0, 10.0)) + (pos3 * make_double3(40.0, 64.0, 40.0)));
    rpos3f0 = (rpos3 * 0.00390625);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_cave_layer_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_cave_layer_1_octave__8)) * 1.0);
    result = (n_8 * 0.8333333333333333);
    output[tid] = result;
}

__global__ void minecraft_cave_layer_51(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_cave_layer_0_octave__8, const int8_t* perm_table_minecraft_cave_layer_1_octave__8, double* output) {
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(2.5, 1.0, 2.5)) + (pos3 * make_double3(10.0, 16.0, 10.0)));
    rpos3f0 = (rpos3 * 0.00390625);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_cave_layer_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_cave_layer_1_octave__8)) * 1.0);
    result = (n_8 * 0.8333333333333333);
    output[tid] = result;
}

__global__ void minecraft_cave_layer_45(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_cave_layer_0_octave__8, const int8_t* perm_table_minecraft_cave_layer_1_octave__8, double* output) {
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(6.0, 6.0, 6.0)) + (pos3 * make_double3(24.0, 96.0, 24.0)));
    rpos3f0 = (rpos3 * 0.00390625);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_cave_layer_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_cave_layer_1_octave__8)) * 1.0);
    result = (n_8 * 0.8333333333333333);
    output[tid] = result;
}

__global__ void minecraft_cave_layer_44(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_cave_layer_0_octave__8, const int8_t* perm_table_minecraft_cave_layer_1_octave__8, double* output) {
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(8.0, 8.0, 8.0)) + (pos3 * make_double3(32.0, 128.0, 32.0)));
    rpos3f0 = (rpos3 * 0.00390625);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_cave_layer_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_cave_layer_1_octave__8)) * 1.0);
    result = (n_8 * 0.8333333333333333);
    output[tid] = result;
}

__global__ void minecraft_cave_layer_8(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_cave_layer_0_octave__8, const int8_t* perm_table_minecraft_cave_layer_1_octave__8, double* output) {
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(0.3, 0.0, 0.3)) + (pos3 * make_double3(1.2, 0.0, 1.2)));
    rpos3f0 = (rpos3 * 0.00390625);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_cave_layer_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_cave_layer_1_octave__8)) * 1.0);
    result = (n_8 * 0.8333333333333333);
    output[tid] = result;
}

__global__ void minecraft_cave_layer_0(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_cave_layer_0_octave__8, const int8_t* perm_table_minecraft_cave_layer_1_octave__8, double* output) {
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(0.25, 0.0, 0.25)) + (pos3 * make_double3(1.0, 0.0, 1.0)));
    rpos3f0 = (rpos3 * 0.00390625);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_cave_layer_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_cave_layer_1_octave__8)) * 1.0);
    result = (n_8 * 0.8333333333333333);
    output[tid] = result;
}

__global__ void minecraft_cave_layer_16(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_cave_layer_0_octave__8, const int8_t* perm_table_minecraft_cave_layer_1_octave__8, double* output) {
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(0.2, 0.0, 0.2)) + (pos3 * make_double3(0.8, 0.0, 0.8)));
    rpos3f0 = (rpos3 * 0.00390625);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_cave_layer_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_cave_layer_1_octave__8)) * 1.0);
    result = (n_8 * 0.8333333333333333);
    output[tid] = result;
}

__global__ void minecraft_cave_layer_46(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_cave_layer_0_octave__8, const int8_t* perm_table_minecraft_cave_layer_1_octave__8, double* output) {
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(4.0, 4.0, 4.0)) + (pos3 * make_double3(16.0, 64.0, 16.0)));
    rpos3f0 = (rpos3 * 0.00390625);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_cave_layer_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_cave_layer_1_octave__8)) * 1.0);
    result = (n_8 * 0.8333333333333333);
    output[tid] = result;
}

__global__ void minecraft_cave_layer_30(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_cave_layer_0_octave__8, const int8_t* perm_table_minecraft_cave_layer_1_octave__8, double* output) {
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(0.1, 0.0, 0.1)) + (pos3 * make_double3(0.4, 0.0, 0.4)));
    rpos3f0 = (rpos3 * 0.00390625);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_cave_layer_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_cave_layer_1_octave__8)) * 1.0);
    result = (n_8 * 0.8333333333333333);
    output[tid] = result;
}

__global__ void minecraft_cave_layer_52(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_cave_layer_0_octave__8, const int8_t* perm_table_minecraft_cave_layer_1_octave__8, double* output) {
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(0.5, 0.5, 0.5)) + (pos3 * make_double3(2.0, 8.0, 2.0)));
    rpos3f0 = (rpos3 * 0.00390625);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_cave_layer_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_cave_layer_1_octave__8)) * 1.0);
    result = (n_8 * 0.8333333333333333);
    output[tid] = result;
}

__global__ void minecraft_cave_layer_50(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_cave_layer_0_octave__8, const int8_t* perm_table_minecraft_cave_layer_1_octave__8, double* output) {
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(5.0, 2.0, 5.0)) + (pos3 * make_double3(20.0, 32.0, 20.0)));
    rpos3f0 = (rpos3 * 0.00390625);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_cave_layer_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_cave_layer_1_octave__8)) * 1.0);
    result = (n_8 * 0.8333333333333333);
    output[tid] = result;
}

__global__ void minecraft_cave_layer_25(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_cave_layer_0_octave__8, const int8_t* perm_table_minecraft_cave_layer_1_octave__8, double* output) {
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(0.15, 0.0, 0.15)) + (pos3 * make_double3(0.6, 0.0, 0.6)));
    rpos3f0 = (rpos3 * 0.00390625);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_cave_layer_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_cave_layer_1_octave__8)) * 1.0);
    result = (n_8 * 0.8333333333333333);
    output[tid] = result;
}

__global__ void minecraft_cave_layer_37(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_cave_layer_0_octave__8, const int8_t* perm_table_minecraft_cave_layer_1_octave__8, double* output) {
    double n_8;
    double3 rpos3;
    double3 rpos3f0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(0.006, 0.0, 0.006)) + (pos3 * make_double3(0.024, 0.0, 0.024)));
    rpos3f0 = (rpos3 * 0.00390625);
    n_8 = ((perlin(rpos3f0, perm_table_minecraft_cave_layer_0_octave__8) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_cave_layer_1_octave__8)) * 1.0);
    result = (n_8 * 0.8333333333333333);
    output[tid] = result;
}

__global__ void minecraft_caves_new_small(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, const double* input_1, const double* input_2, double* output) {
    float _coordinate_263_;
    int32_t _spline_index_264_;
    float _spline_result_f32_265_;
    double _spline_result_f64_266_;
    float _spline_values_267_[165];
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    {
        _spline_values_267_[0] = 1.0f;
        _spline_values_267_[1] = -1.0f;
        _spline_values_267_[2] = 1.0f;
        _spline_values_267_[3] = -1.0f;
        _spline_values_267_[4] = 1.0f;
        _spline_values_267_[5] = -1.0f;
        _spline_values_267_[6] = 1.0f;
        _spline_values_267_[7] = -1.0f;
        _spline_values_267_[8] = 1.0f;
        _spline_values_267_[9] = -1.0f;
        _spline_values_267_[10] = 1.0f;
        _spline_values_267_[11] = -1.0f;
        _spline_values_267_[12] = 1.0f;
        _spline_values_267_[13] = -1.0f;
        _spline_values_267_[14] = 1.0f;
        _spline_values_267_[15] = -1.0f;
        _spline_values_267_[16] = 1.0f;
        _spline_values_267_[17] = -1.0f;
        _spline_values_267_[18] = 1.0f;
        _spline_values_267_[19] = -1.0f;
        _spline_values_267_[20] = 1.0f;
        _spline_values_267_[21] = -1.0f;
        _spline_values_267_[22] = 1.0f;
        _spline_values_267_[23] = -1.0f;
        _spline_values_267_[24] = 1.0f;
        _spline_values_267_[25] = -1.0f;
        _spline_values_267_[26] = 1.0f;
        _spline_values_267_[27] = -1.0f;
        _spline_values_267_[28] = 1.0f;
        _spline_values_267_[29] = -1.0f;
        _spline_values_267_[30] = 1.0f;
        _spline_values_267_[31] = -1.0f;
        _spline_values_267_[32] = 1.0f;
        _spline_values_267_[33] = -1.0f;
        _spline_values_267_[34] = 1.0f;
        _spline_values_267_[35] = -1.0f;
        _spline_values_267_[36] = 1.0f;
        _spline_values_267_[37] = -1.0f;
        _spline_values_267_[38] = 1.0f;
        _spline_values_267_[39] = -1.0f;
        _spline_values_267_[40] = 1.0f;
        _spline_values_267_[41] = -1.0f;
        _spline_values_267_[42] = 1.0f;
        _spline_values_267_[43] = -1.0f;
        _spline_values_267_[44] = 1.0f;
        _spline_values_267_[45] = -1.0f;
        _spline_values_267_[46] = 1.0f;
        _spline_values_267_[47] = -1.0f;
        _spline_values_267_[48] = 1.0f;
        _spline_values_267_[49] = -1.0f;
        _spline_values_267_[50] = 1.0f;
        _spline_values_267_[51] = -1.0f;
        _spline_values_267_[52] = 1.0f;
        _spline_values_267_[53] = -1.0f;
        _spline_values_267_[54] = 1.0f;
        _spline_values_267_[55] = -1.0f;
        _spline_values_267_[56] = 1.0f;
        _spline_values_267_[57] = -1.0f;
        _spline_values_267_[58] = 1.0f;
        _spline_values_267_[59] = -1.0f;
        _spline_values_267_[60] = 1.0f;
        _spline_values_267_[61] = -1.0f;
        _spline_values_267_[62] = 1.0f;
        _spline_values_267_[63] = -1.0f;
        _spline_values_267_[64] = 1.0f;
        _spline_values_267_[65] = -1.0f;
        _spline_values_267_[66] = 1.0f;
        _spline_values_267_[67] = -1.0f;
        _spline_values_267_[68] = 1.0f;
        _spline_values_267_[69] = -1.0f;
        _spline_values_267_[70] = 1.0f;
        _spline_values_267_[71] = -1.0f;
        _spline_values_267_[72] = 1.0f;
        _spline_values_267_[73] = -1.0f;
        _spline_values_267_[74] = 1.0f;
        _spline_values_267_[75] = -1.0f;
        _spline_values_267_[76] = 1.0f;
        _spline_values_267_[77] = -1.0f;
        _spline_values_267_[78] = 1.0f;
        _spline_values_267_[79] = -1.0f;
        _spline_values_267_[80] = 1.0f;
        _spline_values_267_[81] = -1.0f;
        _spline_values_267_[82] = 1.0f;
        _spline_values_267_[83] = -1.0f;
        _spline_values_267_[84] = 1.0f;
        _spline_values_267_[85] = -1.0f;
        _spline_values_267_[86] = 1.0f;
        _spline_values_267_[87] = -1.0f;
        _spline_values_267_[88] = 1.0f;
        _spline_values_267_[89] = -1.0f;
        _spline_values_267_[90] = 1.0f;
        _spline_values_267_[91] = -1.0f;
        _spline_values_267_[92] = 1.0f;
        _spline_values_267_[93] = -1.0f;
        _spline_values_267_[94] = 1.0f;
        _spline_values_267_[95] = -1.0f;
        _spline_values_267_[96] = 1.0f;
        _spline_values_267_[97] = -1.0f;
        _spline_values_267_[98] = 1.0f;
        _spline_values_267_[99] = -1.0f;
        _spline_values_267_[100] = 1.0f;
        _spline_values_267_[101] = -1.0f;
        _spline_values_267_[102] = 1.0f;
        _spline_values_267_[103] = -1.0f;
        _spline_values_267_[104] = 1.0f;
        _spline_values_267_[105] = -1.0f;
        _spline_values_267_[106] = 1.0f;
        _spline_values_267_[107] = -1.0f;
        _spline_values_267_[108] = 1.0f;
        _spline_values_267_[109] = -1.0f;
        _spline_values_267_[110] = 1.0f;
        _spline_values_267_[111] = -1.0f;
        _spline_values_267_[112] = 1.0f;
        _spline_values_267_[113] = -1.0f;
        _spline_values_267_[114] = 1.0f;
        _spline_values_267_[115] = -1.0f;
        _spline_values_267_[116] = 1.0f;
        _spline_values_267_[117] = -1.0f;
        _spline_values_267_[118] = 1.0f;
        _spline_values_267_[119] = -1.0f;
        _spline_values_267_[120] = 1.0f;
        _spline_values_267_[121] = -1.0f;
        _spline_values_267_[122] = 1.0f;
        _spline_values_267_[123] = -1.0f;
        _spline_values_267_[124] = 1.0f;
        _spline_values_267_[125] = -1.0f;
        _spline_values_267_[126] = 1.0f;
        _spline_values_267_[127] = -1.0f;
        _spline_values_267_[128] = 1.0f;
        _spline_values_267_[129] = -1.0f;
        _spline_values_267_[130] = 1.0f;
        _spline_values_267_[131] = -1.0f;
        _spline_values_267_[132] = 1.0f;
        _spline_values_267_[133] = -1.0f;
        _spline_values_267_[134] = 1.0f;
        _spline_values_267_[135] = -1.0f;
        _spline_values_267_[136] = 1.0f;
        _spline_values_267_[137] = -1.0f;
        _spline_values_267_[138] = 1.0f;
        _spline_values_267_[139] = -1.0f;
        _spline_values_267_[140] = 1.0f;
        _spline_values_267_[141] = -1.0f;
        _spline_values_267_[142] = 1.0f;
        _spline_values_267_[143] = -1.0f;
        _spline_values_267_[144] = 1.0f;
        _spline_values_267_[145] = -1.0f;
        _spline_values_267_[146] = 1.0f;
        _spline_values_267_[147] = -1.0f;
        _spline_values_267_[148] = 1.0f;
        _spline_values_267_[149] = -1.0f;
        _spline_values_267_[150] = 1.0f;
        _spline_values_267_[151] = -1.0f;
        _spline_values_267_[152] = 1.0f;
        _spline_values_267_[153] = -1.0f;
        _spline_values_267_[154] = 1.0f;
        _spline_values_267_[155] = -1.0f;
        _spline_values_267_[156] = 1.0f;
        _spline_values_267_[157] = -1.0f;
        _spline_values_267_[158] = 1.0f;
        _spline_values_267_[159] = -1.0f;
        _spline_values_267_[160] = 1.0f;
        _spline_values_267_[161] = -1.0f;
        _spline_values_267_[162] = 1.0f;
        _spline_values_267_[163] = -1.0f;
        _spline_values_267_[164] = 1.0f;
    }
    _coordinate_263_ = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    _spline_index_264_ = binary_search(_spline_coordinates_78_, _coordinate_263_);
    _spline_result_f32_265_ = advanced_hermite(_spline_coordinates_78_, _spline_values_267_, _spline_derivatives_79_, _coordinate_263_, _spline_index_264_);
    _spline_result_f64_266_ = _spline_result_f32_265_;
    result = (_spline_result_f64_266_ + ((input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))] * 1.5) + (input_2[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))] * 0.5)));
    output[tid] = result;
}

__global__ void density_function_Add_38(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, const double* input_1, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = (input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] + (input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * 0.5));
    output[tid] = result;
}

__global__ void minecraft_realism_hills_factor_original(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    output[tid] = result;
}

__global__ void minecraft_realism_mountains_jagged_original(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    output[tid] = result;
}

__global__ void minecraft_caves_old_large(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 660))];
    output[tid] = result;
}

__global__ void minecraft_realism_extreme_hills_factor_divergence_squared(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, const double* input_1, const double* input_2, double* output) {
    double _var_268_;
    double _var_269_;
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    _var_268_ = (input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] + (input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * -1.0));
    _var_269_ = (input_2[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] + (input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * -1.0));
    result = (((_var_268_ * _var_268_) + (_var_269_ * _var_269_)) * 10000.0);
    output[tid] = result;
}

__global__ void minecraft_realism_extreme_hills_factor_original(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    output[tid] = result;
}

__global__ void density_function_Add_29(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, const double* input_1, const double* input_2, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = (((input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * 0.1) + -0.8) + (((input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * 0.0) + 0.0) + ((input_2[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * 0.015) + 0.0)));
    output[tid] = result;
}

__global__ void density_function_Multiply_10(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, const double* input_1, const double* input_2, double* output) {
    double _var_270_;
    double _var_271_;
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    _var_270_ = (input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] + (input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * -1.0));
    _var_271_ = (input_2[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] + (input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * -1.0));
    result = ((((_var_270_ * _var_270_) + (_var_271_ * _var_271_)) * 5000.0) * 0.2);
    output[tid] = result;
}

__global__ void minecraft_new_surface_height_unprocessed_unrivered(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, const double* input_1, const double* input_2, const double* input_3, const double* input_4, const double* input_5, const double* input_6, double* output) {
    float _coordinate_272_;
    float _coordinate_273_;
    float _coordinate_274_;
    float _coordinate_275_;
    float _coordinate_276_;
    float _coordinate_277_;
    float _coordinate_278_;
    int32_t _spline_index_279_;
    int32_t _spline_index_280_;
    int32_t _spline_index_281_;
    int32_t _spline_index_282_;
    int32_t _spline_index_283_;
    int32_t _spline_index_284_;
    int32_t _spline_index_285_;
    float _spline_result_f32_286_;
    double _spline_result_f64_287_;
    float _spline_values_288_[1];
    float _spline_values_289_[1];
    float _spline_values_290_[1];
    float _spline_values_291_[1];
    float _spline_values_292_[1];
    float _spline_values_293_[1];
    float _spline_values_294_[6];
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    {
        _spline_values_288_[0] = 0.0f;
    }
    _coordinate_272_ = input_5[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_279_ = binary_search(_spline_coordinates_80_, _coordinate_272_);
    {
        _spline_values_289_[0] = 0.0f;
    }
    _coordinate_273_ = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_280_ = binary_search(_spline_coordinates_81_, _coordinate_273_);
    {
        _spline_values_290_[0] = 0.0f;
    }
    _coordinate_274_ = input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_281_ = binary_search(_spline_coordinates_82_, _coordinate_274_);
    {
        _spline_values_291_[0] = 0.0f;
    }
    _coordinate_275_ = input_6[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_282_ = binary_search(_spline_coordinates_83_, _coordinate_275_);
    {
        _spline_values_292_[0] = 0.0f;
    }
    _coordinate_276_ = input_3[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_283_ = binary_search(_spline_coordinates_84_, _coordinate_276_);
    {
        _spline_values_293_[0] = 0.0f;
    }
    _coordinate_277_ = input_4[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_284_ = binary_search(_spline_coordinates_85_, _coordinate_277_);
    {
        _spline_values_294_[0] = advanced_hermite(_spline_coordinates_80_, _spline_values_288_, _spline_derivatives_87_, _coordinate_272_, _spline_index_279_);
        _spline_values_294_[1] = advanced_hermite(_spline_coordinates_81_, _spline_values_289_, _spline_derivatives_88_, _coordinate_273_, _spline_index_280_);
        _spline_values_294_[2] = advanced_hermite(_spline_coordinates_82_, _spline_values_290_, _spline_derivatives_89_, _coordinate_274_, _spline_index_281_);
        _spline_values_294_[3] = advanced_hermite(_spline_coordinates_83_, _spline_values_291_, _spline_derivatives_90_, _coordinate_275_, _spline_index_282_);
        _spline_values_294_[4] = advanced_hermite(_spline_coordinates_84_, _spline_values_292_, _spline_derivatives_91_, _coordinate_276_, _spline_index_283_);
        _spline_values_294_[5] = advanced_hermite(_spline_coordinates_85_, _spline_values_293_, _spline_derivatives_92_, _coordinate_277_, _spline_index_284_);
    }
    _coordinate_278_ = input_2[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_285_ = binary_search(_spline_coordinates_86_, _coordinate_278_);
    _spline_result_f32_286_ = advanced_hermite(_spline_coordinates_86_, _spline_values_294_, _spline_derivatives_93_, _coordinate_278_, _spline_index_285_);
    _spline_result_f64_287_ = _spline_result_f32_286_;
    result = _spline_result_f64_287_;
    output[tid] = result;
}

__global__ void minecraft_new_surface_height_unprocessed_rivered(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, const double* input_1, double* output) {
    float _coordinate_295_;
    float _coordinate_296_;
    int32_t _spline_index_297_;
    int32_t _spline_index_298_;
    float _spline_result_f32_299_;
    double _spline_result_f64_300_;
    float _spline_values_301_[1];
    float _spline_values_302_[3];
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    {
        _spline_values_301_[0] = 0.0f;
    }
    _coordinate_295_ = input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_297_ = binary_search(_spline_coordinates_94_, _coordinate_295_);
    {
        _spline_values_302_[0] = advanced_hermite(_spline_coordinates_94_, _spline_values_301_, _spline_derivatives_96_, _coordinate_295_, _spline_index_297_);
        _spline_values_302_[1] = -0.91f;
        _spline_values_302_[2] = advanced_hermite(_spline_coordinates_94_, _spline_values_301_, _spline_derivatives_96_, _coordinate_295_, _spline_index_297_);
    }
    _coordinate_296_ = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_298_ = binary_search(_spline_coordinates_95_, _coordinate_296_);
    _spline_result_f32_299_ = advanced_hermite(_spline_coordinates_95_, _spline_values_302_, _spline_derivatives_97_, _coordinate_296_, _spline_index_298_);
    _spline_result_f64_300_ = _spline_result_f32_299_;
    result = _spline_result_f64_300_;
    output[tid] = result;
}

__global__ void minecraft_realism_mountains_factor_original(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    output[tid] = result;
}

__global__ void density_function_Add_33(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, const double* input_1, const double* input_2, double* output) {
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    result = (((input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * 0.09) + -0.9) + (((input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * 0.05) + 0.0) + ((input_2[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * 0.015) + 0.0)));
    output[tid] = result;
}

__global__ void minecraft_realism_extreme_mountains_slope_height(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, const double* input_1, const double* input_2, const double* input_3, const double* input_4, double* output) {
    float _coordinate_303_;
    float _coordinate_304_;
    float _coordinate_305_;
    float _coordinate_306_;
    int32_t _spline_index_307_;
    int32_t _spline_index_308_;
    int32_t _spline_index_309_;
    int32_t _spline_index_310_;
    float _spline_result_f32_311_;
    float _spline_result_f32_312_;
    double _spline_result_f64_313_;
    double _spline_result_f64_314_;
    float _spline_values_315_[17];
    float _spline_values_316_[1];
    float _spline_values_317_[1];
    float _spline_values_318_[2];
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    {
        _spline_values_315_[0] = 1.0f;
        _spline_values_315_[1] = 0.95238096f;
        _spline_values_315_[2] = 0.90909094f;
        _spline_values_315_[3] = 0.8333333f;
        _spline_values_315_[4] = 0.7692308f;
        _spline_values_315_[5] = 0.71428573f;
        _spline_values_315_[6] = 0.6666667f;
        _spline_values_315_[7] = 0.625f;
        _spline_values_315_[8] = 0.5882353f;
        _spline_values_315_[9] = 0.5555556f;
        _spline_values_315_[10] = 0.5263158f;
        _spline_values_315_[11] = 0.5f;
        _spline_values_315_[12] = 0.47619048f;
        _spline_values_315_[13] = 0.45454547f;
        _spline_values_315_[14] = 0.4347826f;
        _spline_values_315_[15] = 0.41666666f;
        _spline_values_315_[16] = 0.4f;
    }
    _coordinate_303_ = input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_307_ = binary_search(_spline_coordinates_98_, _coordinate_303_);
    _spline_result_f32_311_ = advanced_hermite(_spline_coordinates_98_, _spline_values_315_, _spline_derivatives_102_, _coordinate_303_, _spline_index_307_);
    _spline_result_f64_313_ = _spline_result_f32_311_;
    {
        _spline_values_316_[0] = 0.0f;
    }
    _coordinate_304_ = input_2[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_308_ = binary_search(_spline_coordinates_99_, _coordinate_304_);
    {
        _spline_values_317_[0] = 0.0f;
    }
    _coordinate_305_ = input_2[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_309_ = binary_search(_spline_coordinates_100_, _coordinate_305_);
    {
        _spline_values_318_[0] = advanced_hermite(_spline_coordinates_99_, _spline_values_316_, _spline_derivatives_103_, _coordinate_304_, _spline_index_308_);
        _spline_values_318_[1] = advanced_hermite(_spline_coordinates_100_, _spline_values_317_, _spline_derivatives_104_, _coordinate_305_, _spline_index_309_);
    }
    _coordinate_306_ = input_3[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_310_ = binary_search(_spline_coordinates_101_, _coordinate_306_);
    _spline_result_f32_312_ = advanced_hermite(_spline_coordinates_101_, _spline_values_318_, _spline_derivatives_105_, _coordinate_306_, _spline_index_310_);
    _spline_result_f64_314_ = _spline_result_f32_312_;
    result = ((0.0 + ((input_4[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * 0.2) + 0.3)) + (((input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * _spline_result_f64_313_) * 0.8) + _spline_result_f64_314_));
    output[tid] = result;
}

__global__ void minecraft_realism_hills_slope_height(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const double* input_0, const double* input_1, const double* input_2, const double* input_3, const double* input_4, double* output) {
    float _coordinate_319_;
    float _coordinate_320_;
    float _coordinate_321_;
    float _coordinate_322_;
    float _coordinate_323_;
    float _coordinate_324_;
    int32_t _spline_index_325_;
    int32_t _spline_index_326_;
    int32_t _spline_index_327_;
    int32_t _spline_index_328_;
    int32_t _spline_index_329_;
    int32_t _spline_index_330_;
    float _spline_result_f32_331_;
    float _spline_result_f32_332_;
    double _spline_result_f64_333_;
    double _spline_result_f64_334_;
    float _spline_values_335_[1];
    float _spline_values_336_[1];
    float _spline_values_337_[2];
    float _spline_values_338_[1];
    float _spline_values_339_[1];
    float _spline_values_340_[2];
    double3 rpos3;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * origin_scale) + (pos3 * position_scale));
    {
        _spline_values_335_[0] = 0.0f;
    }
    _coordinate_319_ = input_4[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_325_ = binary_search(_spline_coordinates_106_, _coordinate_319_);
    {
        _spline_values_336_[0] = 0.0f;
    }
    _coordinate_320_ = input_1[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_326_ = binary_search(_spline_coordinates_107_, _coordinate_320_);
    {
        _spline_values_337_[0] = advanced_hermite(_spline_coordinates_106_, _spline_values_335_, _spline_derivatives_112_, _coordinate_319_, _spline_index_325_);
        _spline_values_337_[1] = advanced_hermite(_spline_coordinates_107_, _spline_values_336_, _spline_derivatives_113_, _coordinate_320_, _spline_index_326_);
    }
    _coordinate_321_ = input_4[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_327_ = binary_search(_spline_coordinates_108_, _coordinate_321_);
    _spline_result_f32_331_ = advanced_hermite(_spline_coordinates_108_, _spline_values_337_, _spline_derivatives_114_, _coordinate_321_, _spline_index_327_);
    _spline_result_f64_333_ = _spline_result_f32_331_;
    {
        _spline_values_338_[0] = 0.0f;
    }
    _coordinate_322_ = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_328_ = binary_search(_spline_coordinates_109_, _coordinate_322_);
    {
        _spline_values_339_[0] = 0.0f;
    }
    _coordinate_323_ = input_0[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_329_ = binary_search(_spline_coordinates_110_, _coordinate_323_);
    {
        _spline_values_340_[0] = advanced_hermite(_spline_coordinates_109_, _spline_values_338_, _spline_derivatives_115_, _coordinate_322_, _spline_index_328_);
        _spline_values_340_[1] = advanced_hermite(_spline_coordinates_110_, _spline_values_339_, _spline_derivatives_116_, _coordinate_323_, _spline_index_329_);
    }
    _coordinate_324_ = input_3[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))];
    _spline_index_330_ = binary_search(_spline_coordinates_111_, _coordinate_324_);
    _spline_result_f32_332_ = advanced_hermite(_spline_coordinates_111_, _spline_values_340_, _spline_derivatives_117_, _coordinate_324_, _spline_index_330_);
    _spline_result_f64_334_ = _spline_result_f32_332_;
    result = ((0.0 + ((input_2[((pos3.x + (pos3.y * 5)) + (pos3.z * 5))] * 0.25) + -0.5)) + ((_spline_result_f64_333_ * 0.2) + _spline_result_f64_334_));
    output[tid] = result;
}

__global__ void minecraft_jagged_15(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_jagged_0_octave__16, const int8_t* perm_table_minecraft_jagged_1_octave__16, const int8_t* perm_table_minecraft_jagged_0_octave__15, const int8_t* perm_table_minecraft_jagged_1_octave__15, const int8_t* perm_table_minecraft_jagged_0_octave__14, const int8_t* perm_table_minecraft_jagged_1_octave__14, const int8_t* perm_table_minecraft_jagged_0_octave__13, const int8_t* perm_table_minecraft_jagged_1_octave__13, const int8_t* perm_table_minecraft_jagged_0_octave__12, const int8_t* perm_table_minecraft_jagged_1_octave__12, const int8_t* perm_table_minecraft_jagged_0_octave__11, const int8_t* perm_table_minecraft_jagged_1_octave__11, const int8_t* perm_table_minecraft_jagged_0_octave__10, const int8_t* perm_table_minecraft_jagged_1_octave__10, const int8_t* perm_table_minecraft_jagged_0_octave__9, const int8_t* perm_table_minecraft_jagged_1_octave__9, const int8_t* perm_table_minecraft_jagged_0_octave__8, const int8_t* perm_table_minecraft_jagged_1_octave__8, const int8_t* perm_table_minecraft_jagged_0_octave__7, const int8_t* perm_table_minecraft_jagged_1_octave__7, const int8_t* perm_table_minecraft_jagged_0_octave__6, const int8_t* perm_table_minecraft_jagged_1_octave__6, const int8_t* perm_table_minecraft_jagged_0_octave__5, const int8_t* perm_table_minecraft_jagged_1_octave__5, const int8_t* perm_table_minecraft_jagged_0_octave__4, const int8_t* perm_table_minecraft_jagged_1_octave__4, const int8_t* perm_table_minecraft_jagged_0_octave__3, const int8_t* perm_table_minecraft_jagged_1_octave__3, const int8_t* perm_table_minecraft_jagged_0_octave__2, const int8_t* perm_table_minecraft_jagged_1_octave__2, const int8_t* perm_table_minecraft_jagged_0_octave__1, const int8_t* perm_table_minecraft_jagged_1_octave__1, double* output) {
    double n_1;
    double n_10;
    double n_11;
    double n_12;
    double n_13;
    double n_14;
    double n_15;
    double n_16;
    double n_2;
    double n_3;
    double n_4;
    double n_5;
    double n_6;
    double n_7;
    double n_8;
    double n_9;
    double3 rpos3;
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f10;
    double3 rpos3f11;
    double3 rpos3f12;
    double3 rpos3f13;
    double3 rpos3f14;
    double3 rpos3f15;
    double3 rpos3f2;
    double3 rpos3f3;
    double3 rpos3f4;
    double3 rpos3f5;
    double3 rpos3f6;
    double3 rpos3f7;
    double3 rpos3f8;
    double3 rpos3f9;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(501.0, 0.0, 501.0)) + (pos3 * make_double3(2004.0, 0.0, 2004.0)));
    rpos3f0 = (rpos3 * 0.0000152587890625);
    rpos3f1 = (rpos3 * 0.000030517578125);
    rpos3f2 = (rpos3 * 0.00006103515625);
    rpos3f3 = (rpos3 * 0.0001220703125);
    rpos3f4 = (rpos3 * 0.000244140625);
    rpos3f5 = (rpos3 * 0.00048828125);
    rpos3f6 = (rpos3 * 0.0009765625);
    rpos3f7 = (rpos3 * 0.001953125);
    rpos3f8 = (rpos3 * 0.00390625);
    rpos3f9 = (rpos3 * 0.0078125);
    rpos3f10 = (rpos3 * 0.015625);
    rpos3f11 = (rpos3 * 0.03125);
    rpos3f12 = (rpos3 * 0.0625);
    rpos3f13 = (rpos3 * 0.125);
    rpos3f14 = (rpos3 * 0.25);
    rpos3f15 = (rpos3 * 0.5);
    n_16 = ((perlin(rpos3f0, perm_table_minecraft_jagged_0_octave__16) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__16)) * 0.5000076295109483);
    n_15 = ((perlin(rpos3f1, perm_table_minecraft_jagged_0_octave__15) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__15)) * 0.2500038147554742);
    n_14 = ((perlin(rpos3f2, perm_table_minecraft_jagged_0_octave__14) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__14)) * 0.1250019073777371);
    n_13 = ((perlin(rpos3f3, perm_table_minecraft_jagged_0_octave__13) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__13)) * 0.06250095368886854);
    n_12 = ((perlin(rpos3f4, perm_table_minecraft_jagged_0_octave__12) + perlin((rpos3f4 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__12)) * 0.03125047684443427);
    n_11 = ((perlin(rpos3f5, perm_table_minecraft_jagged_0_octave__11) + perlin((rpos3f5 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__11)) * 0.015625238422217136);
    n_10 = ((perlin(rpos3f6, perm_table_minecraft_jagged_0_octave__10) + perlin((rpos3f6 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__10)) * 0.007812619211108568);
    n_9 = ((perlin(rpos3f7, perm_table_minecraft_jagged_0_octave__9) + perlin((rpos3f7 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__9)) * 0.003906309605554284);
    n_8 = ((perlin(rpos3f8, perm_table_minecraft_jagged_0_octave__8) + perlin((rpos3f8 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__8)) * 0.001953154802777142);
    n_7 = ((perlin(rpos3f9, perm_table_minecraft_jagged_0_octave__7) + perlin((rpos3f9 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__7)) * 0.000976577401388571);
    n_6 = ((perlin(rpos3f10, perm_table_minecraft_jagged_0_octave__6) + perlin((rpos3f10 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__6)) * 0.0004882887006942855);
    n_5 = ((perlin(rpos3f11, perm_table_minecraft_jagged_0_octave__5) + perlin((rpos3f11 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__5)) * 0.00024414435034714275);
    n_4 = ((perlin(rpos3f12, perm_table_minecraft_jagged_0_octave__4) + perlin((rpos3f12 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__4)) * 0.00012207217517357137);
    n_3 = ((perlin(rpos3f13, perm_table_minecraft_jagged_0_octave__3) + perlin((rpos3f13 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__3)) * 0.00006103608758678569);
    n_2 = ((perlin(rpos3f14, perm_table_minecraft_jagged_0_octave__2) + perlin((rpos3f14 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__2)) * 0.000030518043793392844);
    n_1 = ((perlin(rpos3f15, perm_table_minecraft_jagged_0_octave__1) + perlin((rpos3f15 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__1)) * 0.000015259021896696422);
    result = ((((((((((((((((n_16 + n_15) + n_14) + n_13) + n_12) + n_11) + n_10) + n_9) + n_8) + n_7) + n_6) + n_5) + n_4) + n_3) + n_2) + n_1) * 1.568627450980392);
    output[tid] = result;
}

__global__ void minecraft_jagged_7(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_jagged_0_octave__16, const int8_t* perm_table_minecraft_jagged_1_octave__16, const int8_t* perm_table_minecraft_jagged_0_octave__15, const int8_t* perm_table_minecraft_jagged_1_octave__15, const int8_t* perm_table_minecraft_jagged_0_octave__14, const int8_t* perm_table_minecraft_jagged_1_octave__14, const int8_t* perm_table_minecraft_jagged_0_octave__13, const int8_t* perm_table_minecraft_jagged_1_octave__13, const int8_t* perm_table_minecraft_jagged_0_octave__12, const int8_t* perm_table_minecraft_jagged_1_octave__12, const int8_t* perm_table_minecraft_jagged_0_octave__11, const int8_t* perm_table_minecraft_jagged_1_octave__11, const int8_t* perm_table_minecraft_jagged_0_octave__10, const int8_t* perm_table_minecraft_jagged_1_octave__10, const int8_t* perm_table_minecraft_jagged_0_octave__9, const int8_t* perm_table_minecraft_jagged_1_octave__9, const int8_t* perm_table_minecraft_jagged_0_octave__8, const int8_t* perm_table_minecraft_jagged_1_octave__8, const int8_t* perm_table_minecraft_jagged_0_octave__7, const int8_t* perm_table_minecraft_jagged_1_octave__7, const int8_t* perm_table_minecraft_jagged_0_octave__6, const int8_t* perm_table_minecraft_jagged_1_octave__6, const int8_t* perm_table_minecraft_jagged_0_octave__5, const int8_t* perm_table_minecraft_jagged_1_octave__5, const int8_t* perm_table_minecraft_jagged_0_octave__4, const int8_t* perm_table_minecraft_jagged_1_octave__4, const int8_t* perm_table_minecraft_jagged_0_octave__3, const int8_t* perm_table_minecraft_jagged_1_octave__3, const int8_t* perm_table_minecraft_jagged_0_octave__2, const int8_t* perm_table_minecraft_jagged_1_octave__2, const int8_t* perm_table_minecraft_jagged_0_octave__1, const int8_t* perm_table_minecraft_jagged_1_octave__1, double* output) {
    double n_1;
    double n_10;
    double n_11;
    double n_12;
    double n_13;
    double n_14;
    double n_15;
    double n_16;
    double n_2;
    double n_3;
    double n_4;
    double n_5;
    double n_6;
    double n_7;
    double n_8;
    double n_9;
    double3 rpos3;
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f10;
    double3 rpos3f11;
    double3 rpos3f12;
    double3 rpos3f13;
    double3 rpos3f14;
    double3 rpos3f15;
    double3 rpos3f2;
    double3 rpos3f3;
    double3 rpos3f4;
    double3 rpos3f5;
    double3 rpos3f6;
    double3 rpos3f7;
    double3 rpos3f8;
    double3 rpos3f9;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(600.0, 0.0, 600.0)) + (pos3 * make_double3(2400.0, 0.0, 2400.0)));
    rpos3f0 = (rpos3 * 0.0000152587890625);
    rpos3f1 = (rpos3 * 0.000030517578125);
    rpos3f2 = (rpos3 * 0.00006103515625);
    rpos3f3 = (rpos3 * 0.0001220703125);
    rpos3f4 = (rpos3 * 0.000244140625);
    rpos3f5 = (rpos3 * 0.00048828125);
    rpos3f6 = (rpos3 * 0.0009765625);
    rpos3f7 = (rpos3 * 0.001953125);
    rpos3f8 = (rpos3 * 0.00390625);
    rpos3f9 = (rpos3 * 0.0078125);
    rpos3f10 = (rpos3 * 0.015625);
    rpos3f11 = (rpos3 * 0.03125);
    rpos3f12 = (rpos3 * 0.0625);
    rpos3f13 = (rpos3 * 0.125);
    rpos3f14 = (rpos3 * 0.25);
    rpos3f15 = (rpos3 * 0.5);
    n_16 = ((perlin(rpos3f0, perm_table_minecraft_jagged_0_octave__16) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__16)) * 0.5000076295109483);
    n_15 = ((perlin(rpos3f1, perm_table_minecraft_jagged_0_octave__15) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__15)) * 0.2500038147554742);
    n_14 = ((perlin(rpos3f2, perm_table_minecraft_jagged_0_octave__14) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__14)) * 0.1250019073777371);
    n_13 = ((perlin(rpos3f3, perm_table_minecraft_jagged_0_octave__13) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__13)) * 0.06250095368886854);
    n_12 = ((perlin(rpos3f4, perm_table_minecraft_jagged_0_octave__12) + perlin((rpos3f4 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__12)) * 0.03125047684443427);
    n_11 = ((perlin(rpos3f5, perm_table_minecraft_jagged_0_octave__11) + perlin((rpos3f5 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__11)) * 0.015625238422217136);
    n_10 = ((perlin(rpos3f6, perm_table_minecraft_jagged_0_octave__10) + perlin((rpos3f6 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__10)) * 0.007812619211108568);
    n_9 = ((perlin(rpos3f7, perm_table_minecraft_jagged_0_octave__9) + perlin((rpos3f7 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__9)) * 0.003906309605554284);
    n_8 = ((perlin(rpos3f8, perm_table_minecraft_jagged_0_octave__8) + perlin((rpos3f8 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__8)) * 0.001953154802777142);
    n_7 = ((perlin(rpos3f9, perm_table_minecraft_jagged_0_octave__7) + perlin((rpos3f9 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__7)) * 0.000976577401388571);
    n_6 = ((perlin(rpos3f10, perm_table_minecraft_jagged_0_octave__6) + perlin((rpos3f10 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__6)) * 0.0004882887006942855);
    n_5 = ((perlin(rpos3f11, perm_table_minecraft_jagged_0_octave__5) + perlin((rpos3f11 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__5)) * 0.00024414435034714275);
    n_4 = ((perlin(rpos3f12, perm_table_minecraft_jagged_0_octave__4) + perlin((rpos3f12 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__4)) * 0.00012207217517357137);
    n_3 = ((perlin(rpos3f13, perm_table_minecraft_jagged_0_octave__3) + perlin((rpos3f13 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__3)) * 0.00006103608758678569);
    n_2 = ((perlin(rpos3f14, perm_table_minecraft_jagged_0_octave__2) + perlin((rpos3f14 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__2)) * 0.000030518043793392844);
    n_1 = ((perlin(rpos3f15, perm_table_minecraft_jagged_0_octave__1) + perlin((rpos3f15 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__1)) * 0.000015259021896696422);
    result = ((((((((((((((((n_16 + n_15) + n_14) + n_13) + n_12) + n_11) + n_10) + n_9) + n_8) + n_7) + n_6) + n_5) + n_4) + n_3) + n_2) + n_1) * 1.568627450980392);
    output[tid] = result;
}

__global__ void minecraft_jagged_32(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_jagged_0_octave__16, const int8_t* perm_table_minecraft_jagged_1_octave__16, const int8_t* perm_table_minecraft_jagged_0_octave__15, const int8_t* perm_table_minecraft_jagged_1_octave__15, const int8_t* perm_table_minecraft_jagged_0_octave__14, const int8_t* perm_table_minecraft_jagged_1_octave__14, const int8_t* perm_table_minecraft_jagged_0_octave__13, const int8_t* perm_table_minecraft_jagged_1_octave__13, const int8_t* perm_table_minecraft_jagged_0_octave__12, const int8_t* perm_table_minecraft_jagged_1_octave__12, const int8_t* perm_table_minecraft_jagged_0_octave__11, const int8_t* perm_table_minecraft_jagged_1_octave__11, const int8_t* perm_table_minecraft_jagged_0_octave__10, const int8_t* perm_table_minecraft_jagged_1_octave__10, const int8_t* perm_table_minecraft_jagged_0_octave__9, const int8_t* perm_table_minecraft_jagged_1_octave__9, const int8_t* perm_table_minecraft_jagged_0_octave__8, const int8_t* perm_table_minecraft_jagged_1_octave__8, const int8_t* perm_table_minecraft_jagged_0_octave__7, const int8_t* perm_table_minecraft_jagged_1_octave__7, const int8_t* perm_table_minecraft_jagged_0_octave__6, const int8_t* perm_table_minecraft_jagged_1_octave__6, const int8_t* perm_table_minecraft_jagged_0_octave__5, const int8_t* perm_table_minecraft_jagged_1_octave__5, const int8_t* perm_table_minecraft_jagged_0_octave__4, const int8_t* perm_table_minecraft_jagged_1_octave__4, const int8_t* perm_table_minecraft_jagged_0_octave__3, const int8_t* perm_table_minecraft_jagged_1_octave__3, const int8_t* perm_table_minecraft_jagged_0_octave__2, const int8_t* perm_table_minecraft_jagged_1_octave__2, const int8_t* perm_table_minecraft_jagged_0_octave__1, const int8_t* perm_table_minecraft_jagged_1_octave__1, double* output) {
    double n_1;
    double n_10;
    double n_11;
    double n_12;
    double n_13;
    double n_14;
    double n_15;
    double n_16;
    double n_2;
    double n_3;
    double n_4;
    double n_5;
    double n_6;
    double n_7;
    double n_8;
    double n_9;
    double3 rpos3;
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f10;
    double3 rpos3f11;
    double3 rpos3f12;
    double3 rpos3f13;
    double3 rpos3f14;
    double3 rpos3f15;
    double3 rpos3f2;
    double3 rpos3f3;
    double3 rpos3f4;
    double3 rpos3f5;
    double3 rpos3f6;
    double3 rpos3f7;
    double3 rpos3f8;
    double3 rpos3f9;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(500.0, 0.0, 500.0)) + (pos3 * make_double3(2000.0, 0.0, 2000.0)));
    rpos3f0 = (rpos3 * 0.0000152587890625);
    rpos3f1 = (rpos3 * 0.000030517578125);
    rpos3f2 = (rpos3 * 0.00006103515625);
    rpos3f3 = (rpos3 * 0.0001220703125);
    rpos3f4 = (rpos3 * 0.000244140625);
    rpos3f5 = (rpos3 * 0.00048828125);
    rpos3f6 = (rpos3 * 0.0009765625);
    rpos3f7 = (rpos3 * 0.001953125);
    rpos3f8 = (rpos3 * 0.00390625);
    rpos3f9 = (rpos3 * 0.0078125);
    rpos3f10 = (rpos3 * 0.015625);
    rpos3f11 = (rpos3 * 0.03125);
    rpos3f12 = (rpos3 * 0.0625);
    rpos3f13 = (rpos3 * 0.125);
    rpos3f14 = (rpos3 * 0.25);
    rpos3f15 = (rpos3 * 0.5);
    n_16 = ((perlin(rpos3f0, perm_table_minecraft_jagged_0_octave__16) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__16)) * 0.5000076295109483);
    n_15 = ((perlin(rpos3f1, perm_table_minecraft_jagged_0_octave__15) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__15)) * 0.2500038147554742);
    n_14 = ((perlin(rpos3f2, perm_table_minecraft_jagged_0_octave__14) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__14)) * 0.1250019073777371);
    n_13 = ((perlin(rpos3f3, perm_table_minecraft_jagged_0_octave__13) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__13)) * 0.06250095368886854);
    n_12 = ((perlin(rpos3f4, perm_table_minecraft_jagged_0_octave__12) + perlin((rpos3f4 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__12)) * 0.03125047684443427);
    n_11 = ((perlin(rpos3f5, perm_table_minecraft_jagged_0_octave__11) + perlin((rpos3f5 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__11)) * 0.015625238422217136);
    n_10 = ((perlin(rpos3f6, perm_table_minecraft_jagged_0_octave__10) + perlin((rpos3f6 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__10)) * 0.007812619211108568);
    n_9 = ((perlin(rpos3f7, perm_table_minecraft_jagged_0_octave__9) + perlin((rpos3f7 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__9)) * 0.003906309605554284);
    n_8 = ((perlin(rpos3f8, perm_table_minecraft_jagged_0_octave__8) + perlin((rpos3f8 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__8)) * 0.001953154802777142);
    n_7 = ((perlin(rpos3f9, perm_table_minecraft_jagged_0_octave__7) + perlin((rpos3f9 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__7)) * 0.000976577401388571);
    n_6 = ((perlin(rpos3f10, perm_table_minecraft_jagged_0_octave__6) + perlin((rpos3f10 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__6)) * 0.0004882887006942855);
    n_5 = ((perlin(rpos3f11, perm_table_minecraft_jagged_0_octave__5) + perlin((rpos3f11 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__5)) * 0.00024414435034714275);
    n_4 = ((perlin(rpos3f12, perm_table_minecraft_jagged_0_octave__4) + perlin((rpos3f12 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__4)) * 0.00012207217517357137);
    n_3 = ((perlin(rpos3f13, perm_table_minecraft_jagged_0_octave__3) + perlin((rpos3f13 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__3)) * 0.00006103608758678569);
    n_2 = ((perlin(rpos3f14, perm_table_minecraft_jagged_0_octave__2) + perlin((rpos3f14 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__2)) * 0.000030518043793392844);
    n_1 = ((perlin(rpos3f15, perm_table_minecraft_jagged_0_octave__1) + perlin((rpos3f15 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__1)) * 0.000015259021896696422);
    result = ((((((((((((((((n_16 + n_15) + n_14) + n_13) + n_12) + n_11) + n_10) + n_9) + n_8) + n_7) + n_6) + n_5) + n_4) + n_3) + n_2) + n_1) * 1.568627450980392);
    output[tid] = result;
}

__global__ void minecraft_jagged_35(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_jagged_0_octave__16, const int8_t* perm_table_minecraft_jagged_1_octave__16, const int8_t* perm_table_minecraft_jagged_0_octave__15, const int8_t* perm_table_minecraft_jagged_1_octave__15, const int8_t* perm_table_minecraft_jagged_0_octave__14, const int8_t* perm_table_minecraft_jagged_1_octave__14, const int8_t* perm_table_minecraft_jagged_0_octave__13, const int8_t* perm_table_minecraft_jagged_1_octave__13, const int8_t* perm_table_minecraft_jagged_0_octave__12, const int8_t* perm_table_minecraft_jagged_1_octave__12, const int8_t* perm_table_minecraft_jagged_0_octave__11, const int8_t* perm_table_minecraft_jagged_1_octave__11, const int8_t* perm_table_minecraft_jagged_0_octave__10, const int8_t* perm_table_minecraft_jagged_1_octave__10, const int8_t* perm_table_minecraft_jagged_0_octave__9, const int8_t* perm_table_minecraft_jagged_1_octave__9, const int8_t* perm_table_minecraft_jagged_0_octave__8, const int8_t* perm_table_minecraft_jagged_1_octave__8, const int8_t* perm_table_minecraft_jagged_0_octave__7, const int8_t* perm_table_minecraft_jagged_1_octave__7, const int8_t* perm_table_minecraft_jagged_0_octave__6, const int8_t* perm_table_minecraft_jagged_1_octave__6, const int8_t* perm_table_minecraft_jagged_0_octave__5, const int8_t* perm_table_minecraft_jagged_1_octave__5, const int8_t* perm_table_minecraft_jagged_0_octave__4, const int8_t* perm_table_minecraft_jagged_1_octave__4, const int8_t* perm_table_minecraft_jagged_0_octave__3, const int8_t* perm_table_minecraft_jagged_1_octave__3, const int8_t* perm_table_minecraft_jagged_0_octave__2, const int8_t* perm_table_minecraft_jagged_1_octave__2, const int8_t* perm_table_minecraft_jagged_0_octave__1, const int8_t* perm_table_minecraft_jagged_1_octave__1, double* output) {
    double n_1;
    double n_10;
    double n_11;
    double n_12;
    double n_13;
    double n_14;
    double n_15;
    double n_16;
    double n_2;
    double n_3;
    double n_4;
    double n_5;
    double n_6;
    double n_7;
    double n_8;
    double n_9;
    double3 rpos3;
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f10;
    double3 rpos3f11;
    double3 rpos3f12;
    double3 rpos3f13;
    double3 rpos3f14;
    double3 rpos3f15;
    double3 rpos3f2;
    double3 rpos3f3;
    double3 rpos3f4;
    double3 rpos3f5;
    double3 rpos3f6;
    double3 rpos3f7;
    double3 rpos3f8;
    double3 rpos3f9;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(175.0, 0.0, 175.0)) + (pos3 * make_double3(700.0, 0.0, 700.0)));
    rpos3f0 = (rpos3 * 0.0000152587890625);
    rpos3f1 = (rpos3 * 0.000030517578125);
    rpos3f2 = (rpos3 * 0.00006103515625);
    rpos3f3 = (rpos3 * 0.0001220703125);
    rpos3f4 = (rpos3 * 0.000244140625);
    rpos3f5 = (rpos3 * 0.00048828125);
    rpos3f6 = (rpos3 * 0.0009765625);
    rpos3f7 = (rpos3 * 0.001953125);
    rpos3f8 = (rpos3 * 0.00390625);
    rpos3f9 = (rpos3 * 0.0078125);
    rpos3f10 = (rpos3 * 0.015625);
    rpos3f11 = (rpos3 * 0.03125);
    rpos3f12 = (rpos3 * 0.0625);
    rpos3f13 = (rpos3 * 0.125);
    rpos3f14 = (rpos3 * 0.25);
    rpos3f15 = (rpos3 * 0.5);
    n_16 = ((perlin(rpos3f0, perm_table_minecraft_jagged_0_octave__16) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__16)) * 0.5000076295109483);
    n_15 = ((perlin(rpos3f1, perm_table_minecraft_jagged_0_octave__15) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__15)) * 0.2500038147554742);
    n_14 = ((perlin(rpos3f2, perm_table_minecraft_jagged_0_octave__14) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__14)) * 0.1250019073777371);
    n_13 = ((perlin(rpos3f3, perm_table_minecraft_jagged_0_octave__13) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__13)) * 0.06250095368886854);
    n_12 = ((perlin(rpos3f4, perm_table_minecraft_jagged_0_octave__12) + perlin((rpos3f4 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__12)) * 0.03125047684443427);
    n_11 = ((perlin(rpos3f5, perm_table_minecraft_jagged_0_octave__11) + perlin((rpos3f5 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__11)) * 0.015625238422217136);
    n_10 = ((perlin(rpos3f6, perm_table_minecraft_jagged_0_octave__10) + perlin((rpos3f6 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__10)) * 0.007812619211108568);
    n_9 = ((perlin(rpos3f7, perm_table_minecraft_jagged_0_octave__9) + perlin((rpos3f7 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__9)) * 0.003906309605554284);
    n_8 = ((perlin(rpos3f8, perm_table_minecraft_jagged_0_octave__8) + perlin((rpos3f8 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__8)) * 0.001953154802777142);
    n_7 = ((perlin(rpos3f9, perm_table_minecraft_jagged_0_octave__7) + perlin((rpos3f9 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__7)) * 0.000976577401388571);
    n_6 = ((perlin(rpos3f10, perm_table_minecraft_jagged_0_octave__6) + perlin((rpos3f10 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__6)) * 0.0004882887006942855);
    n_5 = ((perlin(rpos3f11, perm_table_minecraft_jagged_0_octave__5) + perlin((rpos3f11 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__5)) * 0.00024414435034714275);
    n_4 = ((perlin(rpos3f12, perm_table_minecraft_jagged_0_octave__4) + perlin((rpos3f12 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__4)) * 0.00012207217517357137);
    n_3 = ((perlin(rpos3f13, perm_table_minecraft_jagged_0_octave__3) + perlin((rpos3f13 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__3)) * 0.00006103608758678569);
    n_2 = ((perlin(rpos3f14, perm_table_minecraft_jagged_0_octave__2) + perlin((rpos3f14 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__2)) * 0.000030518043793392844);
    n_1 = ((perlin(rpos3f15, perm_table_minecraft_jagged_0_octave__1) + perlin((rpos3f15 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__1)) * 0.000015259021896696422);
    result = ((((((((((((((((n_16 + n_15) + n_14) + n_13) + n_12) + n_11) + n_10) + n_9) + n_8) + n_7) + n_6) + n_5) + n_4) + n_3) + n_2) + n_1) * 1.568627450980392);
    output[tid] = result;
}

__global__ void minecraft_jagged_24(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_jagged_0_octave__16, const int8_t* perm_table_minecraft_jagged_1_octave__16, const int8_t* perm_table_minecraft_jagged_0_octave__15, const int8_t* perm_table_minecraft_jagged_1_octave__15, const int8_t* perm_table_minecraft_jagged_0_octave__14, const int8_t* perm_table_minecraft_jagged_1_octave__14, const int8_t* perm_table_minecraft_jagged_0_octave__13, const int8_t* perm_table_minecraft_jagged_1_octave__13, const int8_t* perm_table_minecraft_jagged_0_octave__12, const int8_t* perm_table_minecraft_jagged_1_octave__12, const int8_t* perm_table_minecraft_jagged_0_octave__11, const int8_t* perm_table_minecraft_jagged_1_octave__11, const int8_t* perm_table_minecraft_jagged_0_octave__10, const int8_t* perm_table_minecraft_jagged_1_octave__10, const int8_t* perm_table_minecraft_jagged_0_octave__9, const int8_t* perm_table_minecraft_jagged_1_octave__9, const int8_t* perm_table_minecraft_jagged_0_octave__8, const int8_t* perm_table_minecraft_jagged_1_octave__8, const int8_t* perm_table_minecraft_jagged_0_octave__7, const int8_t* perm_table_minecraft_jagged_1_octave__7, const int8_t* perm_table_minecraft_jagged_0_octave__6, const int8_t* perm_table_minecraft_jagged_1_octave__6, const int8_t* perm_table_minecraft_jagged_0_octave__5, const int8_t* perm_table_minecraft_jagged_1_octave__5, const int8_t* perm_table_minecraft_jagged_0_octave__4, const int8_t* perm_table_minecraft_jagged_1_octave__4, const int8_t* perm_table_minecraft_jagged_0_octave__3, const int8_t* perm_table_minecraft_jagged_1_octave__3, const int8_t* perm_table_minecraft_jagged_0_octave__2, const int8_t* perm_table_minecraft_jagged_1_octave__2, const int8_t* perm_table_minecraft_jagged_0_octave__1, const int8_t* perm_table_minecraft_jagged_1_octave__1, double* output) {
    double n_1;
    double n_10;
    double n_11;
    double n_12;
    double n_13;
    double n_14;
    double n_15;
    double n_16;
    double n_2;
    double n_3;
    double n_4;
    double n_5;
    double n_6;
    double n_7;
    double n_8;
    double n_9;
    double3 rpos3;
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f10;
    double3 rpos3f11;
    double3 rpos3f12;
    double3 rpos3f13;
    double3 rpos3f14;
    double3 rpos3f15;
    double3 rpos3f2;
    double3 rpos3f3;
    double3 rpos3f4;
    double3 rpos3f5;
    double3 rpos3f6;
    double3 rpos3f7;
    double3 rpos3f8;
    double3 rpos3f9;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(300.0, 0.0, 300.0)) + (pos3 * make_double3(1200.0, 0.0, 1200.0)));
    rpos3f0 = (rpos3 * 0.0000152587890625);
    rpos3f1 = (rpos3 * 0.000030517578125);
    rpos3f2 = (rpos3 * 0.00006103515625);
    rpos3f3 = (rpos3 * 0.0001220703125);
    rpos3f4 = (rpos3 * 0.000244140625);
    rpos3f5 = (rpos3 * 0.00048828125);
    rpos3f6 = (rpos3 * 0.0009765625);
    rpos3f7 = (rpos3 * 0.001953125);
    rpos3f8 = (rpos3 * 0.00390625);
    rpos3f9 = (rpos3 * 0.0078125);
    rpos3f10 = (rpos3 * 0.015625);
    rpos3f11 = (rpos3 * 0.03125);
    rpos3f12 = (rpos3 * 0.0625);
    rpos3f13 = (rpos3 * 0.125);
    rpos3f14 = (rpos3 * 0.25);
    rpos3f15 = (rpos3 * 0.5);
    n_16 = ((perlin(rpos3f0, perm_table_minecraft_jagged_0_octave__16) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__16)) * 0.5000076295109483);
    n_15 = ((perlin(rpos3f1, perm_table_minecraft_jagged_0_octave__15) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__15)) * 0.2500038147554742);
    n_14 = ((perlin(rpos3f2, perm_table_minecraft_jagged_0_octave__14) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__14)) * 0.1250019073777371);
    n_13 = ((perlin(rpos3f3, perm_table_minecraft_jagged_0_octave__13) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__13)) * 0.06250095368886854);
    n_12 = ((perlin(rpos3f4, perm_table_minecraft_jagged_0_octave__12) + perlin((rpos3f4 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__12)) * 0.03125047684443427);
    n_11 = ((perlin(rpos3f5, perm_table_minecraft_jagged_0_octave__11) + perlin((rpos3f5 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__11)) * 0.015625238422217136);
    n_10 = ((perlin(rpos3f6, perm_table_minecraft_jagged_0_octave__10) + perlin((rpos3f6 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__10)) * 0.007812619211108568);
    n_9 = ((perlin(rpos3f7, perm_table_minecraft_jagged_0_octave__9) + perlin((rpos3f7 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__9)) * 0.003906309605554284);
    n_8 = ((perlin(rpos3f8, perm_table_minecraft_jagged_0_octave__8) + perlin((rpos3f8 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__8)) * 0.001953154802777142);
    n_7 = ((perlin(rpos3f9, perm_table_minecraft_jagged_0_octave__7) + perlin((rpos3f9 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__7)) * 0.000976577401388571);
    n_6 = ((perlin(rpos3f10, perm_table_minecraft_jagged_0_octave__6) + perlin((rpos3f10 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__6)) * 0.0004882887006942855);
    n_5 = ((perlin(rpos3f11, perm_table_minecraft_jagged_0_octave__5) + perlin((rpos3f11 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__5)) * 0.00024414435034714275);
    n_4 = ((perlin(rpos3f12, perm_table_minecraft_jagged_0_octave__4) + perlin((rpos3f12 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__4)) * 0.00012207217517357137);
    n_3 = ((perlin(rpos3f13, perm_table_minecraft_jagged_0_octave__3) + perlin((rpos3f13 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__3)) * 0.00006103608758678569);
    n_2 = ((perlin(rpos3f14, perm_table_minecraft_jagged_0_octave__2) + perlin((rpos3f14 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__2)) * 0.000030518043793392844);
    n_1 = ((perlin(rpos3f15, perm_table_minecraft_jagged_0_octave__1) + perlin((rpos3f15 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__1)) * 0.000015259021896696422);
    result = ((((((((((((((((n_16 + n_15) + n_14) + n_13) + n_12) + n_11) + n_10) + n_9) + n_8) + n_7) + n_6) + n_5) + n_4) + n_3) + n_2) + n_1) * 1.568627450980392);
    output[tid] = result;
}

__global__ void minecraft_jagged_49(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_jagged_0_octave__16, const int8_t* perm_table_minecraft_jagged_1_octave__16, const int8_t* perm_table_minecraft_jagged_0_octave__15, const int8_t* perm_table_minecraft_jagged_1_octave__15, const int8_t* perm_table_minecraft_jagged_0_octave__14, const int8_t* perm_table_minecraft_jagged_1_octave__14, const int8_t* perm_table_minecraft_jagged_0_octave__13, const int8_t* perm_table_minecraft_jagged_1_octave__13, const int8_t* perm_table_minecraft_jagged_0_octave__12, const int8_t* perm_table_minecraft_jagged_1_octave__12, const int8_t* perm_table_minecraft_jagged_0_octave__11, const int8_t* perm_table_minecraft_jagged_1_octave__11, const int8_t* perm_table_minecraft_jagged_0_octave__10, const int8_t* perm_table_minecraft_jagged_1_octave__10, const int8_t* perm_table_minecraft_jagged_0_octave__9, const int8_t* perm_table_minecraft_jagged_1_octave__9, const int8_t* perm_table_minecraft_jagged_0_octave__8, const int8_t* perm_table_minecraft_jagged_1_octave__8, const int8_t* perm_table_minecraft_jagged_0_octave__7, const int8_t* perm_table_minecraft_jagged_1_octave__7, const int8_t* perm_table_minecraft_jagged_0_octave__6, const int8_t* perm_table_minecraft_jagged_1_octave__6, const int8_t* perm_table_minecraft_jagged_0_octave__5, const int8_t* perm_table_minecraft_jagged_1_octave__5, const int8_t* perm_table_minecraft_jagged_0_octave__4, const int8_t* perm_table_minecraft_jagged_1_octave__4, const int8_t* perm_table_minecraft_jagged_0_octave__3, const int8_t* perm_table_minecraft_jagged_1_octave__3, const int8_t* perm_table_minecraft_jagged_0_octave__2, const int8_t* perm_table_minecraft_jagged_1_octave__2, const int8_t* perm_table_minecraft_jagged_0_octave__1, const int8_t* perm_table_minecraft_jagged_1_octave__1, double* output) {
    double n_1;
    double n_10;
    double n_11;
    double n_12;
    double n_13;
    double n_14;
    double n_15;
    double n_16;
    double n_2;
    double n_3;
    double n_4;
    double n_5;
    double n_6;
    double n_7;
    double n_8;
    double n_9;
    double3 rpos3;
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f10;
    double3 rpos3f11;
    double3 rpos3f12;
    double3 rpos3f13;
    double3 rpos3f14;
    double3 rpos3f15;
    double3 rpos3f2;
    double3 rpos3f3;
    double3 rpos3f4;
    double3 rpos3f5;
    double3 rpos3f6;
    double3 rpos3f7;
    double3 rpos3f8;
    double3 rpos3f9;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(1000.0, 0.0, 1000.0)) + (pos3 * make_double3(4000.0, 0.0, 4000.0)));
    rpos3f0 = (rpos3 * 0.0000152587890625);
    rpos3f1 = (rpos3 * 0.000030517578125);
    rpos3f2 = (rpos3 * 0.00006103515625);
    rpos3f3 = (rpos3 * 0.0001220703125);
    rpos3f4 = (rpos3 * 0.000244140625);
    rpos3f5 = (rpos3 * 0.00048828125);
    rpos3f6 = (rpos3 * 0.0009765625);
    rpos3f7 = (rpos3 * 0.001953125);
    rpos3f8 = (rpos3 * 0.00390625);
    rpos3f9 = (rpos3 * 0.0078125);
    rpos3f10 = (rpos3 * 0.015625);
    rpos3f11 = (rpos3 * 0.03125);
    rpos3f12 = (rpos3 * 0.0625);
    rpos3f13 = (rpos3 * 0.125);
    rpos3f14 = (rpos3 * 0.25);
    rpos3f15 = (rpos3 * 0.5);
    n_16 = ((perlin(rpos3f0, perm_table_minecraft_jagged_0_octave__16) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__16)) * 0.5000076295109483);
    n_15 = ((perlin(rpos3f1, perm_table_minecraft_jagged_0_octave__15) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__15)) * 0.2500038147554742);
    n_14 = ((perlin(rpos3f2, perm_table_minecraft_jagged_0_octave__14) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__14)) * 0.1250019073777371);
    n_13 = ((perlin(rpos3f3, perm_table_minecraft_jagged_0_octave__13) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__13)) * 0.06250095368886854);
    n_12 = ((perlin(rpos3f4, perm_table_minecraft_jagged_0_octave__12) + perlin((rpos3f4 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__12)) * 0.03125047684443427);
    n_11 = ((perlin(rpos3f5, perm_table_minecraft_jagged_0_octave__11) + perlin((rpos3f5 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__11)) * 0.015625238422217136);
    n_10 = ((perlin(rpos3f6, perm_table_minecraft_jagged_0_octave__10) + perlin((rpos3f6 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__10)) * 0.007812619211108568);
    n_9 = ((perlin(rpos3f7, perm_table_minecraft_jagged_0_octave__9) + perlin((rpos3f7 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__9)) * 0.003906309605554284);
    n_8 = ((perlin(rpos3f8, perm_table_minecraft_jagged_0_octave__8) + perlin((rpos3f8 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__8)) * 0.001953154802777142);
    n_7 = ((perlin(rpos3f9, perm_table_minecraft_jagged_0_octave__7) + perlin((rpos3f9 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__7)) * 0.000976577401388571);
    n_6 = ((perlin(rpos3f10, perm_table_minecraft_jagged_0_octave__6) + perlin((rpos3f10 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__6)) * 0.0004882887006942855);
    n_5 = ((perlin(rpos3f11, perm_table_minecraft_jagged_0_octave__5) + perlin((rpos3f11 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__5)) * 0.00024414435034714275);
    n_4 = ((perlin(rpos3f12, perm_table_minecraft_jagged_0_octave__4) + perlin((rpos3f12 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__4)) * 0.00012207217517357137);
    n_3 = ((perlin(rpos3f13, perm_table_minecraft_jagged_0_octave__3) + perlin((rpos3f13 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__3)) * 0.00006103608758678569);
    n_2 = ((perlin(rpos3f14, perm_table_minecraft_jagged_0_octave__2) + perlin((rpos3f14 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__2)) * 0.000030518043793392844);
    n_1 = ((perlin(rpos3f15, perm_table_minecraft_jagged_0_octave__1) + perlin((rpos3f15 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__1)) * 0.000015259021896696422);
    result = ((((((((((((((((n_16 + n_15) + n_14) + n_13) + n_12) + n_11) + n_10) + n_9) + n_8) + n_7) + n_6) + n_5) + n_4) + n_3) + n_2) + n_1) * 1.568627450980392);
    output[tid] = result;
}

__global__ void minecraft_jagged_27(int3 base_pos, int3 dimensions, double3 origin, double3 origin_scale, double3 position_scale, const int8_t* perm_table_minecraft_jagged_0_octave__16, const int8_t* perm_table_minecraft_jagged_1_octave__16, const int8_t* perm_table_minecraft_jagged_0_octave__15, const int8_t* perm_table_minecraft_jagged_1_octave__15, const int8_t* perm_table_minecraft_jagged_0_octave__14, const int8_t* perm_table_minecraft_jagged_1_octave__14, const int8_t* perm_table_minecraft_jagged_0_octave__13, const int8_t* perm_table_minecraft_jagged_1_octave__13, const int8_t* perm_table_minecraft_jagged_0_octave__12, const int8_t* perm_table_minecraft_jagged_1_octave__12, const int8_t* perm_table_minecraft_jagged_0_octave__11, const int8_t* perm_table_minecraft_jagged_1_octave__11, const int8_t* perm_table_minecraft_jagged_0_octave__10, const int8_t* perm_table_minecraft_jagged_1_octave__10, const int8_t* perm_table_minecraft_jagged_0_octave__9, const int8_t* perm_table_minecraft_jagged_1_octave__9, const int8_t* perm_table_minecraft_jagged_0_octave__8, const int8_t* perm_table_minecraft_jagged_1_octave__8, const int8_t* perm_table_minecraft_jagged_0_octave__7, const int8_t* perm_table_minecraft_jagged_1_octave__7, const int8_t* perm_table_minecraft_jagged_0_octave__6, const int8_t* perm_table_minecraft_jagged_1_octave__6, const int8_t* perm_table_minecraft_jagged_0_octave__5, const int8_t* perm_table_minecraft_jagged_1_octave__5, const int8_t* perm_table_minecraft_jagged_0_octave__4, const int8_t* perm_table_minecraft_jagged_1_octave__4, const int8_t* perm_table_minecraft_jagged_0_octave__3, const int8_t* perm_table_minecraft_jagged_1_octave__3, const int8_t* perm_table_minecraft_jagged_0_octave__2, const int8_t* perm_table_minecraft_jagged_1_octave__2, const int8_t* perm_table_minecraft_jagged_0_octave__1, const int8_t* perm_table_minecraft_jagged_1_octave__1, double* output) {
    double n_1;
    double n_10;
    double n_11;
    double n_12;
    double n_13;
    double n_14;
    double n_15;
    double n_16;
    double n_2;
    double n_3;
    double n_4;
    double n_5;
    double n_6;
    double n_7;
    double n_8;
    double n_9;
    double3 rpos3;
    double3 rpos3f0;
    double3 rpos3f1;
    double3 rpos3f10;
    double3 rpos3f11;
    double3 rpos3f12;
    double3 rpos3f13;
    double3 rpos3f14;
    double3 rpos3f15;
    double3 rpos3f2;
    double3 rpos3f3;
    double3 rpos3f4;
    double3 rpos3f5;
    double3 rpos3f6;
    double3 rpos3f7;
    double3 rpos3f8;
    double3 rpos3f9;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dimensions.x * dimensions.y * dimensions.z) return;
    int ux__ = tid % dimensions.x;
    int uy__ = (tid / dimensions.x) % dimensions.y;
    int uz__ = tid / (dimensions.x * dimensions.y);
    int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);
    double result = 0.0;
    rpos3 = ((origin * make_double3(250.0, 0.0, 250.0)) + (pos3 * make_double3(1000.0, 0.0, 1000.0)));
    rpos3f0 = (rpos3 * 0.0000152587890625);
    rpos3f1 = (rpos3 * 0.000030517578125);
    rpos3f2 = (rpos3 * 0.00006103515625);
    rpos3f3 = (rpos3 * 0.0001220703125);
    rpos3f4 = (rpos3 * 0.000244140625);
    rpos3f5 = (rpos3 * 0.00048828125);
    rpos3f6 = (rpos3 * 0.0009765625);
    rpos3f7 = (rpos3 * 0.001953125);
    rpos3f8 = (rpos3 * 0.00390625);
    rpos3f9 = (rpos3 * 0.0078125);
    rpos3f10 = (rpos3 * 0.015625);
    rpos3f11 = (rpos3 * 0.03125);
    rpos3f12 = (rpos3 * 0.0625);
    rpos3f13 = (rpos3 * 0.125);
    rpos3f14 = (rpos3 * 0.25);
    rpos3f15 = (rpos3 * 0.5);
    n_16 = ((perlin(rpos3f0, perm_table_minecraft_jagged_0_octave__16) + perlin((rpos3f0 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__16)) * 0.5000076295109483);
    n_15 = ((perlin(rpos3f1, perm_table_minecraft_jagged_0_octave__15) + perlin((rpos3f1 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__15)) * 0.2500038147554742);
    n_14 = ((perlin(rpos3f2, perm_table_minecraft_jagged_0_octave__14) + perlin((rpos3f2 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__14)) * 0.1250019073777371);
    n_13 = ((perlin(rpos3f3, perm_table_minecraft_jagged_0_octave__13) + perlin((rpos3f3 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__13)) * 0.06250095368886854);
    n_12 = ((perlin(rpos3f4, perm_table_minecraft_jagged_0_octave__12) + perlin((rpos3f4 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__12)) * 0.03125047684443427);
    n_11 = ((perlin(rpos3f5, perm_table_minecraft_jagged_0_octave__11) + perlin((rpos3f5 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__11)) * 0.015625238422217136);
    n_10 = ((perlin(rpos3f6, perm_table_minecraft_jagged_0_octave__10) + perlin((rpos3f6 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__10)) * 0.007812619211108568);
    n_9 = ((perlin(rpos3f7, perm_table_minecraft_jagged_0_octave__9) + perlin((rpos3f7 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__9)) * 0.003906309605554284);
    n_8 = ((perlin(rpos3f8, perm_table_minecraft_jagged_0_octave__8) + perlin((rpos3f8 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__8)) * 0.001953154802777142);
    n_7 = ((perlin(rpos3f9, perm_table_minecraft_jagged_0_octave__7) + perlin((rpos3f9 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__7)) * 0.000976577401388571);
    n_6 = ((perlin(rpos3f10, perm_table_minecraft_jagged_0_octave__6) + perlin((rpos3f10 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__6)) * 0.0004882887006942855);
    n_5 = ((perlin(rpos3f11, perm_table_minecraft_jagged_0_octave__5) + perlin((rpos3f11 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__5)) * 0.00024414435034714275);
    n_4 = ((perlin(rpos3f12, perm_table_minecraft_jagged_0_octave__4) + perlin((rpos3f12 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__4)) * 0.00012207217517357137);
    n_3 = ((perlin(rpos3f13, perm_table_minecraft_jagged_0_octave__3) + perlin((rpos3f13 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__3)) * 0.00006103608758678569);
    n_2 = ((perlin(rpos3f14, perm_table_minecraft_jagged_0_octave__2) + perlin((rpos3f14 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__2)) * 0.000030518043793392844);
    n_1 = ((perlin(rpos3f15, perm_table_minecraft_jagged_0_octave__1) + perlin((rpos3f15 * 1.0181268882175227), perm_table_minecraft_jagged_1_octave__1)) * 0.000015259021896696422);
    result = ((((((((((((((((n_16 + n_15) + n_14) + n_13) + n_12) + n_11) + n_10) + n_9) + n_8) + n_7) + n_6) + n_5) + n_4) + n_3) + n_2) + n_1) * 1.568627450980392);
    output[tid] = result;
}

