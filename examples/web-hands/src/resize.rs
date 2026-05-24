use std::{collections::HashMap, num::NonZeroUsize};

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Rgb<T> {
	pub r: T,
	pub g: T,
	pub b: T
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Rgba<T> {
	pub r: T,
	pub g: T,
	pub b: T,
	pub a: T
}

trait Filter {
	const RADIUS: f32;

	fn kernel(&self, x: f32) -> f32;
}

struct BilinearFilter;
impl Filter for BilinearFilter {
	const RADIUS: f32 = 1.0;
	#[inline(always)]
	fn kernel(&self, x: f32) -> f32 {
		(1.0 - x.abs()).max(0.0)
	}
}

#[derive(Debug, Clone)]
struct CoeffsLine {
	start: usize,
	coeffs: Box<[f32]>
}

#[derive(Debug)]
struct Scale {
	w1: NonZeroUsize,
	h1: NonZeroUsize,
	coeffs_w: Vec<CoeffsLine>,
	coeffs_h: Vec<CoeffsLine>
}

impl Scale {
	pub fn new(source_width: usize, source_height: usize, dest_width: usize, dest_height: usize) -> Self {
		let (source_width, source_height) = (NonZeroUsize::new(source_width).unwrap(), NonZeroUsize::new(source_height).unwrap());
		if dest_width == 0 || dest_height == 0 {
			panic!();
		}

		let mut recycled_coeffs = HashMap::new();
		recycled_coeffs.reserve(dest_width.max(dest_height));

		let coeffs_w = Self::calc_coeffs(source_width, dest_width, &mut recycled_coeffs);
		let coeffs_h = if source_height == source_width && dest_height == dest_width {
			coeffs_w.clone()
		} else {
			Self::calc_coeffs(source_height, dest_height, &mut recycled_coeffs)
		};

		Self {
			w1: source_width,
			h1: source_height,
			coeffs_w,
			coeffs_h
		}
	}

	fn calc_coeffs(s1: NonZeroUsize, s2: usize, recycled_coeffs: &mut HashMap<(usize, [u8; 4], [u8; 4]), Box<[f32]>>) -> Vec<CoeffsLine> {
		Self::calc_coeffs_dispatch(s1, s2, BilinearFilter, recycled_coeffs)
	}

	#[inline(never)]
	fn calc_coeffs_dispatch<F: Filter>(
		s1: NonZeroUsize,
		s2: usize,
		filter: F,
		recycled_coeffs: &mut HashMap<(usize, [u8; 4], [u8; 4]), Box<[f32]>>
	) -> Vec<CoeffsLine> {
		let ratio = s1.get() as f64 / s2 as f64;
		let filter_scale = ratio.max(1.);
		let filter_radius = (f64::from(F::RADIUS) * filter_scale).ceil();
		let mut res = Vec::new();
		res.reserve_exact(s2);
		for x2 in 0..s2 {
			let x1 = (x2 as f64 + 0.5) * ratio - 0.5;
			let start = (x1 - filter_radius).ceil() as isize;
			let start = start.min(s1.get() as isize - 1).max(0) as usize;
			let end = (x1 + filter_radius).floor() as isize;
			let end = (end.min(s1.get() as isize - 1).max(0) as usize).max(start);
			let sum: f64 = (start..=end)
				.map(|i| f64::from(filter.kernel(((i as f64 - x1) / filter_scale) as f32)))
				.sum();
			let key = (end - start, (filter_scale as f32).to_ne_bytes(), (start as f32 - x1 as f32).to_ne_bytes());
			let coeffs = if let Some(k) = recycled_coeffs.get(&key) {
				k.clone()
			} else {
				let tmp = (start..=end)
					.map(|i| {
						let n = ((i as f64 - x1) / filter_scale) as f32;
						(f64::from(filter.kernel(n.min(F::RADIUS).max(-F::RADIUS))) / sum) as f32
					})
					.collect::<Box<[_]>>();
				recycled_coeffs.reserve(1);
				recycled_coeffs.insert(key, tmp.clone());
				tmp
			};
			res.push(CoeffsLine { start, coeffs });
		}
		res
	}

	#[inline(always)]
	fn w2(&self) -> usize {
		self.coeffs_w.len()
	}

	#[inline(always)]
	fn h2(&self) -> usize {
		self.coeffs_h.len()
	}
}

pub trait PixelFormat {
	type InputPixel: Copy;
	type OutputPixel: Default + Copy;
	type Accumulator: Copy;

	fn new() -> Self::Accumulator;
	fn add(&self, acc: &mut Self::Accumulator, inp: Self::InputPixel, coeff: f32);
	fn add_acc(acc: &mut Self::Accumulator, inp: Self::Accumulator, coeff: f32);
	fn into_pixel(&self, acc: Self::Accumulator) -> Self::OutputPixel;
}

pub struct RGBA8ToF32 {
	scale: f32,
	offset: f32
}

impl RGBA8ToF32 {
	pub const ZERO_TO_1: Self = RGBA8ToF32::new(255.0, 0.0);

	const fn new(scale: f32, offset: f32) -> Self {
		Self { scale, offset }
	}
}

impl PixelFormat for RGBA8ToF32 {
	type InputPixel = Rgba<u8>;
	type OutputPixel = Rgb<f32>;
	type Accumulator = Rgb<f32>;

	#[inline(always)]
	fn new() -> Self::Accumulator {
		Rgb { r: 0.0, g: 0.0, b: 0.0 }
	}

	#[inline(always)]
	fn add(&self, acc: &mut Self::Accumulator, inp: Rgba<u8>, coeff: f32) {
		acc.r += inp.r as f32 * coeff;
		acc.g += inp.g as f32 * coeff;
		acc.b += inp.b as f32 * coeff;
	}

	#[inline(always)]
	fn add_acc(acc: &mut Self::Accumulator, inp: Self::Accumulator, coeff: f32) {
		acc.r += inp.r * coeff;
		acc.g += inp.g * coeff;
		acc.b += inp.b * coeff;
	}

	#[inline(always)]
	fn into_pixel(&self, acc: Self::Accumulator) -> Rgb<f32> {
		Rgb {
			r: (acc.r + self.offset) / self.scale,
			g: (acc.g + self.offset) / self.scale,
			b: (acc.b + self.offset) / self.scale
		}
	}
}

#[derive(Debug)]
pub struct Resizer<F: PixelFormat> {
	scale: Scale,
	format: F,
	tmp: Vec<F::Accumulator>
}

impl<F: PixelFormat> Resizer<F> {
	#[inline]
	pub fn new(source_width: usize, source_height: usize, dest_width: usize, dest_height: usize, pixel_format: F) -> Self {
		Self {
			scale: Scale::new(source_width, source_height, dest_width, dest_height),
			format: pixel_format,
			tmp: Vec::new()
		}
	}

	pub(crate) fn resize_strided(&mut self, src: &[F::InputPixel], src_stride: usize, dst: &mut [F::OutputPixel], dst_stride: usize, dst_x_offset: usize) {
		if self.scale.w1.get() > src_stride
			|| self.scale.w2() > dst_stride
			|| src.len() < (src_stride * self.scale.h1.get()) + self.scale.w1.get() - src_stride
			|| dst.len() != ((dst_stride * self.scale.h2()) + self.scale.w2() - dst_stride) + (dst_x_offset * 2)
		{
			panic!();
		}

		let w2 = self.scale.w2();

		self.tmp.clear();
		self.tmp.reserve_exact(w2 * self.scale.h1.get());

		let mut src_rows = src.chunks(src_stride);
		for (dst, row) in dst.chunks_exact_mut(dst_stride).zip(&self.scale.coeffs_h) {
			let end = w2 * (row.start + row.coeffs.len());
			while self.tmp.len() < end {
				let row = src_rows.next().unwrap();
				let format = &self.format;
				self.tmp.extend(self.scale.coeffs_w.iter().map(|col| {
					let in_px = row.get(col.start..col.start + col.coeffs.len()).unwrap_or_default();
					let mut accum = F::new();
					for (coeff, in_px) in col.coeffs.iter().copied().zip(in_px.iter().copied()) {
						format.add(&mut accum, in_px, coeff);
					}
					accum
				}));
			}

			let tmp_row_start = &self.tmp.get(w2 * row.start..).unwrap_or_default();
			for (col, dst_px) in dst.iter_mut().skip(dst_x_offset).take(w2).enumerate() {
				let mut accum = F::new();
				for (coeff, other_row) in row.coeffs.iter().copied().zip(tmp_row_start.iter().copied().skip(col).step_by(w2)) {
					F::add_acc(&mut accum, other_row, coeff);
				}
				*dst_px = self.format.into_pixel(accum);
			}
		}
	}
}

pub struct LetterboxedResizer<F: PixelFormat> {
	resizer: Resizer<F>,
	pub y_shift: usize,
	pub x_shift: usize,
	stride: usize
}

impl<F: PixelFormat> LetterboxedResizer<F> {
	#[inline]
	pub fn new(source_width: usize, source_height: usize, dest_width: usize, dest_height: usize, pixel_format: F) -> Self {
		let r = (dest_width as f32 / source_width as f32).min(dest_height as f32 / source_height as f32);
		let (w, h) = ((source_width as f32 * r).round() as usize, (source_height as f32 * r).round() as usize);

		let (x_shift, y_shift) = if w == dest_width { (0, (dest_height - h) / 2) } else { ((dest_width - w) / 2, 0) };

		Self {
			resizer: Resizer::new(source_width, source_height, w, h, pixel_format),
			y_shift,
			x_shift,
			stride: dest_width
		}
	}

	pub fn resize(&mut self, src: &[F::InputPixel], dst: &mut [F::OutputPixel]) {
		let shift = self.y_shift * self.stride;
		let out_buffer = &mut dst[shift..shift + (self.resizer.scale.h2() * self.stride)];
		self.resizer
			.resize_strided(src, self.resizer.scale.w1.get(), out_buffer, self.stride, self.x_shift);
	}
}
