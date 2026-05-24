use serde::Serialize;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy, PartialEq, PartialOrd, Serialize)]
pub struct Point {
	pub x: f32,
	pub y: f32
}

impl Point {
	pub const fn new(x: f32, y: f32) -> Point {
		Self { x, y }
	}
}

#[derive(Serialize, Default, Debug, Clone)]
pub struct Detection {
	pub score: f32,
	pub center: Point,
	pub width: f32,
	pub height: f32,
	pub min: Point,
	pub max: Point
}

impl Detection {
	pub fn new(score: f32, center: Point, width: f32, height: f32) -> Self {
		Self {
			score,
			center,
			width,
			height,
			min: Point::new(center.x - width / 2., center.y - height / 2.),
			max: Point::new(center.x + width / 2., center.y + height / 2.)
		}
	}

	#[inline(always)]
	fn x1(&self) -> f32 {
		self.min.x
	}
	#[inline(always)]
	fn x2(&self) -> f32 {
		self.max.x
	}
	#[inline(always)]
	fn y1(&self) -> f32 {
		self.min.y
	}
	#[inline(always)]
	fn y2(&self) -> f32 {
		self.max.y
	}

	pub fn to_roi(&mut self, src_dimensions: (u32, u32), y_offset: f32, expansion: f32) {
		let long_side = (self.width * src_dimensions.0 as f32).max(self.height * src_dimensions.1 as f32);
		self.center = Point::new(self.center.x * src_dimensions.0 as f32, (self.center.y + (y_offset * self.height)) * src_dimensions.1 as f32);
		self.width = long_side * expansion;
		self.height = long_side * expansion;

		self.min = Point::new(self.center.x - self.width / 2., self.center.y - self.height / 2.);
		self.max = Point::new(self.center.x + self.width / 2., self.center.y + self.height / 2.);
	}

	#[inline]
	fn intersection(&self, other: &Detection) -> f32 {
		let (x1, y1) = (self.x1().max(other.x1()), self.y1().max(other.y1()));
		let (x2, y2) = (self.x2().min(other.x2()), self.y2().min(other.y2()));
		(x2 - x1).max(0.0) * (y2 - y1).max(0.0)
	}

	#[inline]
	fn union_with_intersection(&self, other: &Detection, intersection: f32) -> f32 {
		let aa = (self.x2() - self.x1()).abs() * (self.y2() - self.y1()).abs();
		let ba = (other.x2() - other.x1()).abs() * (other.y2() - other.y1()).abs();
		aa + ba - intersection
	}

	#[inline]
	pub fn iou(&self, other: &Detection) -> f32 {
		let i = self.intersection(other);
		i / self.union_with_intersection(other, i)
	}
}

pub struct NonMaximumSuppression {
	retains: Vec<bool>,
	retained_locations: Vec<usize>,
	indexed_scores: Vec<(usize, f32)>,
	max_results: usize,
	min_suppression_threshold: f32
}

impl NonMaximumSuppression {
	#[inline]
	pub fn new(mut max_results: usize, min_suppression_threshold: f32) -> Self {
		if max_results == 0 {
			max_results = usize::MAX;
		}

		Self {
			retains: Vec::new(),
			retained_locations: Vec::new(),
			indexed_scores: Vec::new(),
			max_results,
			min_suppression_threshold
		}
	}

	pub fn process_inplace(&mut self, detections: &mut Vec<Detection>) {
		self.indexed_scores.clear();
		self.indexed_scores.reserve_exact(detections.len());
		for i in 0..detections.len() {
			self.indexed_scores.push((i, detections[i].score));
		}

		self.non_max_suppression_inplace(detections)
	}

	fn non_max_suppression_inplace(&mut self, detections: &mut Vec<Detection>) {
		self.retains.clear();
		self.retains.reserve_exact(detections.len());
		unsafe {
			self.retains.set_len(detections.len());
			self.retains.fill(false);
		}

		self.retained_locations.clear();
		for &(index, _score) in &self.indexed_scores {
			let location = &detections[index];
			let mut suppressed = false;

			for &retained_location in &self.retained_locations {
				let similarity = location.iou(&detections[retained_location]);
				if similarity > self.min_suppression_threshold {
					suppressed = true;
					break;
				}
			}

			if !suppressed {
				self.retains[index] = true;
				self.retained_locations.push(index);
				if self.retained_locations.len() >= self.max_results {
					break;
				}
			}
		}

		let mut iter = self.retains.iter();
		detections.retain(|_| *iter.next().unwrap())
	}
}
