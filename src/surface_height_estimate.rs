use crate::orchestration::{self, PermutationTables};
use crate::mathf64::Vec3;


pub fn estimate_surface_height(permutation_tables: &PermutationTables, column_pos: u64) -> i32 {
    let x = (column_pos & 0xFFFFFFFF) as u32 as i32;
    let z = ((column_pos >> 32) & 0xFFFFFFFF) as u32 as i32;
    let k = -64 as i32;
    let CELL_SIZE_Y = 16;
    
    let origin = Vec3::new(x as f64, k as f64, z as f64);
    let densities = orchestration::orchestrate_initial_density_without_jaggedness(origin, permutation_tables);
    
    // take every 4th density value
    // then only keep the first quarter of them
    let densities_new = densities.iter().step_by(4).take(densities.len() / 16).cloned().collect::<Vec<_>>();
    
    assert_eq!(densities.len() / 16, densities_new.len());
    for (index, value) in densities_new.into_iter().enumerate().rev() {
        if value > 0.390625 {
            return (index as i32 * CELL_SIZE_Y + k);
        }
    }

    
    return i32::MAX;
}

pub fn fill_surface_height_estimates(permutation_tables: &PermutationTables, chunk_x: i32, chunk_z: i32, buffer: &mut [i32; 16]) {
    let k = -64 as i32;
    let CELL_SIZE_Y = 16;
    let x = chunk_x << 4;
    let z = chunk_z << 4;
    let origin = Vec3::new(x as f64, k as f64, z as f64);
    let densities = orchestration::orchestrate_initial_density_without_jaggedness(origin, permutation_tables);
    
    for i in 0..4 {
        for j in 0..4 {
            let surface_height = find_surface_height(i, j, k, CELL_SIZE_Y, &densities.as_slice());
            buffer[(i * 4 + j) as usize] = surface_height;
        }
    }

}

fn find_surface_height(i: i32, j: i32, k: i32, cell_size_y: i32, densities: &[f64]) -> i32 {
    let y_length = densities.len() / 16;
    let mut local_index = y_length as i32;
    // let mut local_index = j; // x
    // local_index += ; // y
    // local_index += i * y_length as i32; // z
    while local_index > 0 {
        local_index -= 1;
        let global_index = i * y_length as i32 * 4 + local_index * 4 + j;
        let value = densities[global_index as usize];
        if value > 0.390625 {
            return (local_index as i32 * cell_size_y + k);
        }
    }
    i32::MAX
}