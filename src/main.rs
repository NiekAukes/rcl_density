#![allow(warnings)]

use std::{
    io::Write,
    sync::Arc,
    thread,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use crate::{mathf64::Vec3, utilsf64::set_perlin_seed};

mod density_function;
mod gpu_orchestrator;
pub mod math;
pub mod mathf64;
mod orchestration;
pub mod perlin;
pub mod random;
//mod test_server;
pub mod utils;
pub mod utilsf64;
pub mod xoroshiro;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if let Some(pos) = args.iter().position(|a| a == "--test-server") {
        let addr = args
            .get(pos + 1)
            .map(|s| s.as_str())
            .unwrap_or("127.0.0.1:9876");
        //test_server::run(addr);
        return;
    }

    if args.iter().any(|a| a == "--benchmark") {
        let output = args
            .iter()
            .position(|a| a == "--output")
            .and_then(|i| args.get(i + 1))
            .map(|s| s.clone())
            .unwrap_or_else(|| "chunk_benchmark_rcl.csv".to_string());

        let builder = thread::Builder::new()
            .name("BenchmarkThread".to_string())
            .stack_size(1024 * 1024 * 10);

        let handle = builder
            .spawn(move || {
                run_benchmark(&output);
            })
            .unwrap();

        handle.join().unwrap();
        return;
    }

    if args.iter().any(|a| a == "--gpu") {
        run_gpu();
        return;
    }

    // Default: compute one chunk
    let builder = thread::Builder::new()
        .name("DensityThread".to_string())
        .stack_size(1024 * 1024 * 10);

    let handle = builder
        .spawn(|| {
            let mut rnd = xoroshiro::Xoroshiro128PlusPlusRandom::new(214140, 12411);
            let x = orchestration_seeded(
                rnd.next_long(),
                Vec3 {
                    x: rnd.next_int_bound(1000) as f64,
                    y: rnd.next_int_bound(1000) as f64,
                    z: rnd.next_int_bound(1000) as f64,
                },
            );
            let _ = x.final_density;
        })
        .unwrap();

    handle.join().unwrap();
}

fn run_benchmark(output: &str) {
    // Fixed world seed — same source as the default run
    let mut rnd = xoroshiro::Xoroshiro128PlusPlusRandom::new(214140, 12411);
    let world_seed = rnd.next_long();

    // Initialise permutation tables once for the world seed
    let perm_tables = set_perlin_seed(world_seed);

    let field = (-16, 16);

    println!(
        "Benchmarking {}x{} chunks...",
        field.1 - field.0,
        field.1 - field.0
    );

    struct Record {
        chunk_x: i32,
        chunk_z: i32,
        duration_ms: f64,
        timestamp_ms: u128,
    }

    let mut records: Vec<Record> = Vec::with_capacity(64 * 64);

    for cx in field.0..field.1 {
        for cz in field.0..field.1 {
            let origin = Vec3 {
                x: (cx * 16) as f64,
                y: 0.0,
                z: (cz * 16) as f64,
            };

            let timestamp_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis();

            let t0 = Instant::now();
            let _result = orchestration::orchestration(origin, perm_tables);
            let duration_ms = t0.elapsed().as_secs_f64() * 1000.0;

            records.push(Record {
                chunk_x: cx,
                chunk_z: cz,
                duration_ms,
                timestamp_ms,
            });
        }
    }

    // Write CSV
    let file = std::fs::File::create(output).expect("Failed to create output file");
    let mut writer = std::io::BufWriter::new(file);
    writeln!(writer, "chunk_x,chunk_z,duration_ms,timestamp_ms").unwrap();
    for r in &records {
        writeln!(
            writer,
            "{},{},{:.3},{}",
            r.chunk_x, r.chunk_z, r.duration_ms, r.timestamp_ms
        )
        .unwrap();
    }

    let durations: Vec<f64> = records.iter().map(|r| r.duration_ms).collect();
    let n = durations.len() as f64;
    let mean = durations.iter().sum::<f64>() / n;
    let mut sorted = durations.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let p95 = sorted[(sorted.len() as f64 * 0.95) as usize];

    println!("Done. {} chunks", records.len());
    println!("  mean   : {:.2} ms", mean);
    println!("  median : {:.2} ms", median);
    println!("  min    : {:.2} ms", sorted[0]);
    println!("  max    : {:.2} ms", sorted[sorted.len() - 1]);
    println!("  p95    : {:.2} ms", p95);
    println!("Saved → {}", output);
}

/// Initialize the Perlin sampler with the given seed before computing density
pub fn orchestration_seeded(seed: i64, origin: Vec3) -> orchestration::OrchestrationOutput {
    let perm_tables = set_perlin_seed(seed);
    orchestration::orchestration(origin, perm_tables)
}

fn run_gpu() {
    let mut rnd = xoroshiro::Xoroshiro128PlusPlusRandom::new(214140, 12411);
    let seed = rnd.next_long();
    let origin = Vec3 {
        x: rnd.next_int_bound(1000) as f64,
        y: rnd.next_int_bound(1000) as f64,
        z: rnd.next_int_bound(1000) as f64,
    };
    let perm_tables = orchestration::make_permutation_tables(seed);
    println!("Compiling GPU shaders...");
    let gpu = gpu_orchestrator::GpuOrchestrator_final_density::new();
    //let result = gpu.orchestrate(origin, &perm_tables);
    // run 1000 iterations at once to amortize throughput.
    let gpu = Arc::new(gpu);
    let perm_tables = Arc::new(perm_tables);
    println!("Running GPU orchestration...");
    let begin = std::time::Instant::now();
    let result = gpu.orchestrate(origin, &perm_tables);
    let duration = begin.elapsed();
    println!("GPU orchestration completed in {:.2?}", duration);
    println!("GPU density computed: {} elements", result.len());
    println!("First 10 density values: {:?}", &result[..10]);

    let begin = std::time::Instant::now();
    //let cpu_result = orchestration_seeded(seed, origin).final_density;
    let builder = thread::Builder::new()
        .name("DensityThread".to_string())
        .stack_size(1024 * 1024 * 100);

    let handle = builder
        .spawn(move || orchestration_seeded(seed, origin).final_density)
        .unwrap();
    let cpu_result = handle.join().unwrap();

    let duration = begin.elapsed();
    println!("CPU orchestration completed in {:.2?}", duration);
    println!("CPU density computed: {} elements", cpu_result.len());
    println!("First 10 density values: {:?}", &cpu_result[..10]);
}
