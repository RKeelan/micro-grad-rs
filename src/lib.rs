mod scalar;
pub use scalar::Scalar;

pub fn arrange(start: f64, stop: f64, step: f64) -> impl Iterator<Item = f64> {
    let count = ((stop - start) / step).ceil() as usize;
    (0..count).map(move |i| start + step * i as f64)
}
