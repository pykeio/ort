use std::io::{self, Read};

use hmac_sha256::Hash;

pub fn hex_str_to_bytes(c: impl AsRef<[u8]>) -> Vec<u8> {
	fn nibble(c: u8) -> u8 {
		match c {
			b'A'..=b'F' => c - b'A' + 10,
			b'a'..=b'f' => c - b'a' + 10,
			b'0'..=b'9' => c - b'0',
			_ => panic!()
		}
	}

	c.as_ref().chunks(2).map(|n| (nibble(n[0]) << 4) | nibble(n[1])).collect()
}

pub fn bytes_to_hex_str(bytes: &[u8]) -> String {
	const HEX_CHARS: &[u8] = b"0123456789abcdef";
	let mut s = String::new();
	for b in bytes {
		s.push(HEX_CHARS[(*b >> 4) as usize] as char);
		s.push(HEX_CHARS[(*b & 0xf) as usize] as char);
	}
	s
}

pub struct VerifyReader<R> {
	reader: R,
	state: Hash
}

impl<R: Read> VerifyReader<R> {
	pub fn new(reader: R) -> Self {
		Self { reader, state: Hash::new() }
	}

	pub fn finalize(mut self) -> io::Result<([u8; 32], R)> {
		// For whatever reason, `lzma-rust2` leaves some data unread at the end of the stream. Make sure all of the stream is
		// consumed so we get the correct hash.
		io::copy(&mut self, &mut io::sink())?;

		Ok((self.state.finalize(), self.reader))
	}
}

impl<R: Read> Read for VerifyReader<R> {
	fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
		let b = self.reader.read(buf)?;
		self.state.update(&buf[..b]);
		Ok(b)
	}
}
