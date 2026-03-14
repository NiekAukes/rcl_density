#[derive(Copy, Clone, Debug)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn dot(&self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn fract(&self) -> Vec3 {
        Vec3::new(self.x.fract(), self.y.fract(), self.z.fract())
    }
    pub fn floor(&self) -> Vec3 {
        Vec3::new(self.x.floor(), self.y.floor(), self.z.floor())
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl std::ops::Mul for Vec3 {
    type Output = Self;
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, scalar: f32) -> Vec3 {
        Vec3::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}

impl std::ops::Mul<i32> for Vec3 {
    type Output = Self;
    fn mul(self, scalar: i32) -> Vec3 {
        Vec3::new(
            self.x * scalar as f32,
            self.y * scalar as f32,
            self.z * scalar as f32,
        )
    }
}

impl std::ops::Div for Vec3 {
    type Output = Self;
    fn div(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x / other.x, self.y / other.y, self.z / other.z)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Pos3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Pos3 {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

impl std::ops::Mul<i32> for Pos3 {
    type Output = Self;
    fn mul(self, scalar: i32) -> Pos3 {
        Pos3::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}

impl std::ops::Add for Pos3 {
    type Output = Self;
    fn add(self, other: Pos3) -> Pos3 {
        Pos3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl std::ops::Sub for Pos3 {
    type Output = Self;
    fn sub(self, other: Pos3) -> Pos3 {
        Pos3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl std::ops::Mul for Pos3 {
    type Output = Self;
    fn mul(self, other: Pos3) -> Pos3 {
        Pos3::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

impl std::ops::Add<i32> for Pos3 {
    type Output = Self;
    fn add(self, scalar: i32) -> Pos3 {
        Pos3::new(self.x + scalar, self.y + scalar, self.z + scalar)
    }
}

impl std::ops::Add<Pos3> for Vec3 {
    type Output = Self;
    fn add(self, other: Pos3) -> Vec3 {
        Vec3::new(
            self.x + other.x as f32,
            self.y + other.y as f32,
            self.z + other.z as f32,
        )
    }
}

impl std::ops::Sub<Pos3> for Vec3 {
    type Output = Self;
    fn sub(self, other: Pos3) -> Vec3 {
        Vec3::new(
            self.x - other.x as f32,
            self.y - other.y as f32,
            self.z - other.z as f32,
        )
    }
}

impl std::ops::Mul<Pos3> for Vec3 {
    type Output = Self;
    fn mul(self, other: Pos3) -> Vec3 {
        Vec3::new(
            self.x * other.x as f32,
            self.y * other.y as f32,
            self.z * other.z as f32,
        )
    }
}

impl std::ops::Mul<Vec3> for Pos3 {
    type Output = Vec3;
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.x as f32 * other.x,
            self.y as f32 * other.y,
            self.z as f32 * other.z,
        )
    }
}

#[inline(always)]
pub fn as_index(pos: Pos3, size_x: i32, size_y: i32) -> usize {
    (pos.z * size_y * size_x + pos.y * size_x + pos.x) as usize
}

pub fn pow(base: f32, exp: f32) -> f32 {
    base.powf(exp)
}

pub struct Iter3D {
    x: i32,
    y: i32,
    z: i32,
    pub mx: i32,
    pub my: i32,
    pub mz: i32,
}

impl Iterator for Iter3D {
    type Item = Pos3;
    fn next(&mut self) -> Option<Pos3> {
        if self.z >= self.mz {
            return None;
        }
        let current = Pos3::new(self.x, self.y, self.z);
        self.x += 1;
        if self.x >= self.mx {
            self.x = 0;
            self.y += 1;
            if self.y >= self.my {
                self.y = 0;
                self.z += 1;
            }
        }
        Some(current)
    }
}

pub fn iter_3d(mx: i32, my: i32, mz: i32) -> Iter3D {
    Iter3D {
        x: 0,
        y: 0,
        z: 0,
        mx,
        my,
        mz,
    }
}
