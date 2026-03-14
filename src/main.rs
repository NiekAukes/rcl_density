#![allow(warnings)]

use std::thread;

use crate::math::{Pos3, Vec3};

mod density_function;
pub mod math;
mod orchestration;
mod test_server;
pub mod utils;

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
            let x = orchestration::orchestration(Vec3 {
                x: 1.0,
                y: 1.0,
                z: 1.0,
            });

            return x.11;
        })
        .unwrap();

    let x = handle.join().unwrap();
    //println!("Density at (0,0,0): {:?}", x);
}
