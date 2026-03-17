#![allow(warnings)]

pub mod math;
pub mod utils;
pub mod perlin;
pub mod random;
mod density_function;
mod orchestration;

pub use math::{Vec3, Pos3};
pub use utils::set_perlin_seed;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}


/// Compute a full 16x256x16 density chunk with the given seed and origin.
/// Returns 13 output arrays (one for each density function output).
// pub fn compute_density_chunk(
//     seed: u32,
//     origin: Vec3,
// ) -> [Box<[f32; 65536]>; 13] {
//     let (a, b, c, d, e, f, g, h, i, j, k, l, m) = orchestration_seeded(seed, origin);
//     [a, b, c, d, e, f, g, h, i, j, k, l, m]
// }

/// Sample density at a single point given a seed and origin.
/// Returns the main final density value (the 12th output, index 11).
pub fn sample_density_at(seed: u32, origin: Vec3) -> f32 {
    let outputs = orchestration_seeded(seed, origin);
    outputs.continents[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

/// Initialize the Perlin sampler with the given seed before computing density
pub fn orchestration_seeded(
    seed: u32,
    origin: Vec3,
) -> orchestration::OrchestrationOutput {
    set_perlin_seed(seed);
    orchestration::orchestration(origin)
}