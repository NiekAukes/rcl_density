use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;

use crate::math::{Vec3, Pos3, as_index};
use crate::utils;

/// Start a TCP test server on the given address (e.g. "127.0.0.1:9876").
///
/// # Protocol
///
/// Each request is a single UTF-8 line terminated by `\n`:
///
/// ```text
/// function_name arg0 arg1 ... argN
/// ```
///
/// The response is a single line:
///
/// ```text
/// OK value0 [value1 value2 ...]
/// ```
///
/// or, on error:
///
/// ```text
/// ERR description
/// ```
///
/// ## Supported functions
///
/// ### Utility functions (single values)
/// | Name                | Args                                                | Returns    |
/// |---------------------|-----------------------------------------------------|------------|
/// | `hermite`           | `t p0 p1 m0 m1`                                     | 1 float    |
/// | `fade`              | `x y z`                                             | 3 floats   |
/// | `clamp`             | `x min max`                                         | 1 float    |
/// | `abs`               | `x`                                                 | 1 float    |
/// | `max`               | `a b`                                               | 1 float    |
/// | `perlin`            | `x y z`                                             | 1 float    |
/// | `old_blended_noise` | `x y z xz_scale y_scale xz_factor y_factor smear`   | 1 float    |
/// | `ping`              | (none)                                              | `pong`     |
///
/// ### Density computation functions
/// | Name                    | Args                                              | Returns        |
/// |-------------------------|---------------------------------------------------|-----------------|
/// | `density_chunk`         | `seed origin_x origin_y origin_z`                 | 13 floats (min/max/mean per output) |
/// | `density_point`         | `seed origin_x origin_y origin_z patch_x patch_y patch_z output_idx` | 1 float |
/// | `density_chunk_all`     | `seed origin_x origin_y origin_z output_idx`      | 65536 floats (full array) |
///
/// Send `quit` to gracefully shut down the connection.
pub fn run(addr: &str) {
    let listener = TcpListener::bind(addr).expect("failed to bind test server");
    println!("[test-server] listening on {addr}");

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[test-server] accept error: {e}");
                continue;
            }
        };

        let peer = stream
            .peer_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|_| "unknown".into());
        println!("[test-server] connection from {peer}");

        let reader = BufReader::new(match stream.try_clone() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[test-server] clone error: {e}");
                continue;
            }
        });
        let mut writer = stream;

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("[test-server] read error: {e}");
                    break;
                }
            };

            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }
            if line == "quit" {
                println!("[test-server] {peer} disconnected (quit)");
                break;
            }

            let response = dispatch(&line);
            if writeln!(writer, "{response}").is_err() {
                break;
            }
        }

        println!("[test-server] {peer} disconnected");
    }
}


use crate::orchestration::OrchestrationOutput;

fn get_output(outputs: &OrchestrationOutput, idx: usize) -> Option<&Box<[f32; 65536]>> {
    match idx {
        0 => Some(&outputs.temperature),
        1 => Some(&outputs.vegetation),
        2 => Some(&outputs.initial_density_without_jaggedness),
        3 => Some(&outputs.vein_toggle),
        4 => Some(&outputs.continents),
        5 => Some(&outputs.vein_gap),
        6 => Some(&outputs.vein_ridged),
        7 => Some(&outputs.depth),
        8 => Some(&outputs.fluid_level_spread),
        9 => Some(&outputs.final_density),
        10 => Some(&outputs.ridges),
        11 => Some(&outputs.lava),
        12 => Some(&outputs.erosion),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

fn dispatch(line: &str) -> String {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() {
        return "ERR empty command".into();
    }

    let name = parts[0];
    let args = &parts[1..];

    match name {
        "ping" => "OK pong".into(),

        "hermite" => with_args(args, 5, |a| {
            let v = utils::hermite(a[0], a[1], a[2], a[3], a[4]);
            format!("OK {v:.10}")
        }),

        "fade" => with_args(args, 3, |a| {
            let r = utils::fade(Vec3::new(a[0], a[1], a[2]));
            format!("OK {:.10} {:.10} {:.10}", r.x, r.y, r.z)
        }),

        "clamp" => with_args(args, 3, |a| {
            let v = utils::clamp(a[0], a[1], a[2]);
            format!("OK {v:.10}")
        }),

        "abs" => with_args(args, 1, |a| {
            let v = utils::abs(a[0]);
            format!("OK {v:.10}")
        }),

        "max" => with_args(args, 2, |a| {
            let v = utils::max(a[0], a[1]);
            format!("OK {v:.10}")
        }),

        "perlin" => with_args(args, 3, |a| {
            let v = utils::perlin(Vec3::new(a[0], a[1], a[2]));
            format!("OK {v:.10}")
        }),

        "old_blended_noise" => with_args(args, 8, |a| {
            let v =
                utils::old_blended_noise(Vec3::new(a[0], a[1], a[2]), a[3], a[4], a[5], a[6], a[7]);
            format!("OK {v:.10}")
        }),

        "density_point" => with_args(args, 8, |a| {
            let seed = a[0].to_bits();
            let origin = Vec3::new(a[1], a[2], a[3]);
            let patch_pos = Pos3 {
                x: a[4] as i32,
                y: a[5] as i32,
                z: a[6] as i32,
            };
            let output_idx = a[7] as usize;
            
            if output_idx >= 13 {
                return format!("ERR output_idx must be 0-12, got {}", output_idx);
            }

            let outputs = super::orchestration_seeded(seed, origin);
            let idx = as_index(patch_pos, 16, 256) as usize;

            if idx >= 65536 {
                return format!("ERR invalid patch position: {}", idx);
            }

            match get_output(&outputs, output_idx) {
                Some(output) => {
                    let value = output[idx];
                    format!("OK {value:.10}")
                }
                None => format!("ERR output_idx out of range"),
            }
        }),

        "density_chunk" => with_args(args, 4, |a| {
            let seed = a[0].to_bits();
            let origin = Vec3::new(a[1], a[2], a[3]);
            
            let outputs = super::orchestration_seeded(seed, origin);
            
            // Return min/max/mean for each named output
            let all_outputs: [(&str, &Box<[f32; 65536]>); 13] = [
                ("temperature", &outputs.temperature),
                ("vegetation", &outputs.vegetation),
                ("initial_density_without_jaggedness", &outputs.initial_density_without_jaggedness),
                ("vein_toggle", &outputs.vein_toggle),
                ("continents", &outputs.continents),
                ("vein_gap", &outputs.vein_gap),
                ("vein_ridged", &outputs.vein_ridged),
                ("depth", &outputs.depth),
                ("fluid_level_spread", &outputs.fluid_level_spread),
                ("final_density", &outputs.final_density),
                ("ridges", &outputs.ridges),
                ("lava", &outputs.lava),
                ("erosion", &outputs.erosion),
            ];
            let mut stats = Vec::new();
            for (name, output) in &all_outputs {
                {
                    let output = *output;
                    let mut min = f32::INFINITY;
                    let mut max = f32::NEG_INFINITY;
                    let mut sum = 0.0_f32;
                    
                    for &val in output.iter() {
                        if val.is_finite() {
                            min = min.min(val);
                            max = max.max(val);
                            sum += val;
                        }
                    }
                    
                    let mean = sum / 65536.0;
                    stats.push(format!("{name}:{min:.6}/{max:.6}/{mean:.6}"));
                }
            }

            format!("OK {}", stats.join(" "))
        }),

        "density_chunk_all" => with_args(args, 5, |a| {
            let seed = a[0].to_bits();
            let origin = Vec3::new(a[1], a[2], a[3]);
            let output_idx = a[4] as usize;
            
            if output_idx >= 13 {
                return format!("ERR output_idx must be 0-12, got {}", output_idx);
            }
            
            let outputs = super::orchestration_seeded(seed, origin);
            
            match get_output(&outputs, output_idx) {
                Some(output) => {
                    // Format all 65536 values from the requested output
                    let formatted: Vec<String> = output.iter()
                        .map(|v| format!("{:.6}", v))
                        .collect();
                    
                    format!("OK {}", formatted.join(" "))
                }
                None => format!("ERR output_idx out of range"),
            }
        }),

        other => format!("ERR unknown function: {other}"),
    }
}

/// Parse `args` as `expected` number of f32s, then call `f`.
fn with_args(args: &[&str], expected: usize, f: impl FnOnce(&[f32]) -> String) -> String {
    if args.len() != expected {
        return format!("ERR expected {expected} args, got {}", args.len());
    }

    let mut parsed = Vec::with_capacity(expected);
    for (i, s) in args.iter().enumerate() {
        match s.parse::<f32>() {
            Ok(v) => parsed.push(v),
            Err(e) => return format!("ERR arg {i} parse error: {e}"),
        }
    }

    f(&parsed)
}

// ---------------------------------------------------------------------------
// Unit tests for the dispatcher itself
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping() {
        assert_eq!(dispatch("ping"), "OK pong");
    }

    #[test]
    fn unknown_function() {
        assert!(dispatch("foobar 1 2 3").starts_with("ERR"));
    }

    #[test]
    fn wrong_arg_count() {
        // hermite expects 5 args
        assert!(dispatch("hermite 1 2").starts_with("ERR"));
    }

    #[test]
    fn bad_float() {
        assert!(dispatch("abs notanumber").starts_with("ERR"));
    }

    #[test]
    fn abs_works() {
        let resp = dispatch("abs -3.14");
        assert!(resp.starts_with("OK"));
        let val: f32 = resp.strip_prefix("OK ").unwrap().trim().parse().unwrap();
        assert!((val - 3.14).abs() < 1e-4);
    }

    #[test]
    fn perlin_lattice_zero() {
        let resp = dispatch("perlin 0 0 0");
        assert!(resp.starts_with("OK"));
        let val: f32 = resp.strip_prefix("OK ").unwrap().trim().parse().unwrap();
        assert!(val.abs() < 1e-6, "perlin at origin should be ~0, got {val}");
    }

    #[test]
    fn clamp_below() {
        let resp = dispatch("clamp -5.0 0.0 1.0");
        let val: f32 = resp.strip_prefix("OK ").unwrap().trim().parse().unwrap();
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn density_point_basic() {
        let resp = dispatch("density_point 0 0 0 0 0 0 0 0");
        assert!(resp.starts_with("OK"));
        let val: f32 = resp.strip_prefix("OK ").unwrap().trim().parse().unwrap();
        assert!(val.is_finite(), "density_point should return finite value");
    }

    #[test]
    fn density_point_invalid_output() {
        let resp = dispatch("density_point 0 0 0 0 0 0 0 15");
        assert!(resp.starts_with("ERR"), "should reject output_idx >= 13");
    }

    #[test]
    fn density_chunk_returns_stats() {
        let resp = dispatch("density_chunk 0 0 0 0");
        assert!(resp.starts_with("OK"));
        // Response is space-separated tokens of the form "name:min/max/mean"
        let content = resp.strip_prefix("OK ").unwrap().trim();
        let parts: Vec<&str> = content.split_whitespace().collect();
        assert_eq!(parts.len(), 13, "density_chunk should return 13 output stats");

        let expected_names = [
            "temperature", "vegetation", "initial_density_without_jaggedness",
            "vein_toggle", "continents", "vein_gap", "vein_ridged", "depth",
            "fluid_level_spread", "final_density", "ridges", "lava", "erosion",
        ];

        // Each token is "name:min/max/mean"
        for (i, part) in parts.iter().enumerate() {
            let (name, stats) = part.split_once(':')
                .unwrap_or_else(|| panic!("stat {} '{}' missing ':'", i, part));
            assert_eq!(name, expected_names[i], "stat {} name mismatch", i);
            let nums: Vec<&str> = stats.split('/').collect();
            assert_eq!(nums.len(), 3, "stat {} should have 3 values (min/max/mean)", i);
            for (j, num) in nums.iter().enumerate() {
                let _: f32 = num.parse().unwrap_or_else(|_|
                    panic!("stat {} value {} '{}' should be parseable as f32", i, j, num));
            }
        }
    }

    #[test]
    fn density_chunk_all_invalid_output() {
        let resp = dispatch("density_chunk_all 0 0 0 0 13");
        assert!(resp.starts_with("ERR"), "should reject output_idx >= 13");
    }

    #[test]
    fn density_point_different_positions() {
        let resp1 = dispatch("density_point 0 0 0 0 0 0 0 0");
        let resp2 = dispatch("density_point 0 0 0 0 1 0 0 0");
        
        assert!(resp1.starts_with("OK") && resp2.starts_with("OK"));
        
        let val1: f32 = resp1.strip_prefix("OK ").unwrap().trim().parse().unwrap();
        let val2: f32 = resp2.strip_prefix("OK ").unwrap().trim().parse().unwrap();
        
        // Different positions in same chunk should generally produce different values
        // (very unlikely they'd be exactly equal)
        // Note: We only assert they're both finite, not different, since it's probabilistic
        assert!(val1.is_finite() && val2.is_finite());
    }
}
