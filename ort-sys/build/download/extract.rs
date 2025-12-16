//! Note that this tar reader is only functional enough to read tar files produced by Deno's @std/tar, version 0.1.8.
//! Zero unsafe allowed here. Since we verify the file on-the-fly, the file should be considered untrusted.

#![forbid(unsafe_code)]

use std::{
	fs::File,
	io::{self, BufWriter, Read},
	path::{Component, Path, PathBuf},
	str
};

use crate::error::{Error, ResultExt};

fn parse_octal(bytes: &[u8]) -> io::Result<u64> {
	let s = bytes.iter().take_while(|v| **v != 0).map(|v| *v as char).collect::<String>();
	u64::from_str_radix(s.trim().trim_matches('\0'), 8).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad octal data"))
}

fn parse_str(bytes: &[u8]) -> &str {
	let pos = bytes.iter().position(|v| *v == 0).unwrap_or(bytes.len());
	str::from_utf8(&bytes[..pos]).unwrap_or("")
}

#[derive(Debug)]
struct TarHeader {
	data: [u8; 512]
}

impl TarHeader {
	fn read_from<R: Read>(r: &mut R) -> io::Result<Option<Self>> {
		let mut data = [0; 512];
		if let Err(e) = r.read_exact(&mut data) {
			match e.kind() {
				io::ErrorKind::UnexpectedEof if data.iter().all(|b| *b == 0) => return Ok(None),
				_ => return Err(e)
			}
		}

		if data.iter().all(|b| *b == 0) {
			return Ok(None);
		}
		Ok(Some(Self { data }))
	}

	fn slice(&self, offset: usize, len: usize) -> &[u8] {
		&self.data[offset..offset + len]
	}

	fn checksum(&self) -> u64 {
		let mut sum = 0;
		for (i, v) in self.data.iter().enumerate() {
			if !(148..156).contains(&i) {
				sum += *v as u64;
			} else {
				sum += 32;
			}
		}
		sum
	}

	fn path(&self) -> String {
		let prefix = parse_str(self.slice(345, 155));
		let name = parse_str(self.slice(0, 100));
		if !prefix.is_empty() { format!("{prefix}/{name}") } else { name.to_string() }
	}
}

pub fn extract_tgz<R: Read>(reader: &mut R, output: &Path) -> Result<(), Error> {
	let mut tar = lzma_rust2::Lzma2Reader::new(reader, 1 << 26, None);
	let mut pad_container = [0; 512];

	loop {
		let Some(header) = TarHeader::read_from(&mut tar).with_context(|| "Failed to read tar entry header")? else {
			// actually ends with 1024 zero bytes, so read the remaining 512 to ensure we consume all the data so the hash is
			// correct.
			tar.read_exact(&mut pad_container)?;
			return Ok(());
		};

		if parse_octal(header.slice(148, 8)).with_context(|| "Failed to read tar entry checksum")? != header.checksum() {
			return Err(Error::new("tar entry checksum does not match"));
		}

		if header.data[156] != b'0' {
			// directory; skip
			continue;
		}

		// only allow entries below the root
		let mut subpath = PathBuf::new();
		for comp in PathBuf::from(header.path()).components() {
			match comp {
				Component::Prefix(_) | Component::RootDir => {
					return Err(Error::new("invalid entry path"));
				}
				Component::CurDir => {
					continue;
				}
				Component::ParentDir => {
					subpath.pop();
				}
				Component::Normal(x) => {
					subpath.push(x);
				}
			}
		}

		let path = output.join(subpath);
		if let Some(parent) = path.parent() {
			let _ = std::fs::create_dir_all(parent);
		}

		let size = parse_octal(header.slice(124, 12)).with_context(|| "Failed to read tar entry size")?;
		let mut file = BufWriter::new(File::create_new(&path).with_context(|| format!("Failed to create file '{}'", path.display()))?);

		let copied_bytes = io::copy(&mut tar.by_ref().take(size), &mut file).with_context(|| format!("Failed to extract to '{}'", path.display()))?;
		assert_eq!(size, copied_bytes, "did not copy full entry");

		let padding_bytes = size.next_multiple_of(512) - size;
		if padding_bytes != 0 {
			tar.read_exact(&mut pad_container[..padding_bytes as usize])
				.with_context(|| "Failed to skip padding bytes")?;
		}
	}
}
