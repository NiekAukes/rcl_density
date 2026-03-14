use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;

use crate::math::Vec3;
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
}
