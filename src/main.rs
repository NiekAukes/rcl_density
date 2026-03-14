#![allow(warnings)]

use std::thread;

use crate::{math::{Pos3, Vec3}, utils::set_perlin_seed};


mod density_function;
pub mod math;
mod orchestration;
mod test_server;
pub mod utils;
pub mod perlin;
pub mod random;
fn main() {
    // Launch the test server if --test-server is passed.
    // Usage: cargo run -- --test-server [addr:port]
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--test-server") {
        let addr = args
            .get(pos + 1)
            .map(|s| s.as_str())
            .unwrap_or("127.0.0.1:9876");
        test_server::run(addr);
        return;
    }
    // let x = orchestration::orchestration(Vec3 {
    //     x: 0.0,
    //     y: 0.0,
    //     z: 0.0,
    // });
    let builder = thread::Builder::new()
        .name("DensityThread".to_string())
        .stack_size(1024 * 1024 * 10);

    let handle = builder
        .spawn(|| {
            // Use seed 0 for now; in real usage, this would be the world seed
            let x = orchestration_seeded(0, Vec3 {
                x: 1.0,
                y: 1.0,
                z: 1.0,
            });

            return x.final_density;
        })
        .unwrap();

    let x = handle.join().unwrap();
    //println!("Density at (0,0,0): {:?}", x);
}

/// Initialize the Perlin sampler with the given seed before computing density
pub fn orchestration_seeded(
    seed: u32,
    origin: Vec3,
) -> orchestration::OrchestrationOutput {
    set_perlin_seed(seed);
    orchestration::orchestration(origin)
}