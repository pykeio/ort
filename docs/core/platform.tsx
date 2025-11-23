import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

export type Platform = 'windows' | 'linux' | 'macos' | 'web' | 'ios' | 'android';
export type Arch = 'x86' | 'x64' | 'arm' | 'arm64' | 'web';
export type Triple = { os: Platform, arch: Arch }; // really a double but whatever
export type TripleFilter = Partial<Triple>;

export const PLATFORMS_WITH_BINARIES: Triple[] = [
	{ os: 'windows', arch: 'x64' },
	{ os: 'windows', arch: 'arm64' },
	{ os: 'macos', arch: 'arm64' },
	{ os: 'linux', arch: 'x64' },
	{ os: 'linux', arch: 'arm64' },
	{ os: 'ios', arch: 'arm64' },
	{ os: 'android', arch: 'arm64' },
	{ os: 'web', arch: 'web' }
];
export const PLATFORMS_SUPPORTED: Triple[] = [
	{ os: 'windows', arch: 'x86' },
	{ os: 'windows', arch: 'x64' },
	{ os: 'windows', arch: 'arm' },
	{ os: 'windows', arch: 'arm64' },
	{ os: 'linux', arch: 'x86' },
	{ os: 'linux', arch: 'x64' },
	{ os: 'linux', arch: 'arm' },
	{ os: 'linux', arch: 'arm64' },
	{ os: 'macos', arch: 'x64' },
	{ os: 'macos', arch: 'arm64' },
	{ os: 'ios', arch: 'arm64' },
	{ os: 'android', arch: 'arm64' },
	{ os: 'web', arch: 'web' }
];

export const PLATFORM_NOTES: [TripleFilter, React.ReactNode][] = [
	[ { os: 'windows' }, 'Prebuilt binaries require a recent version of Windows 11 and Visual Studio 2022 (≥ 17.14)' ],
	[ { os: 'linux' }, 'Prebuilt binaries require glibc ≥ 2.39 & libstdc++ ≥ 13.2 (Ubuntu ≥ 24.04, Debian ≥ 13 ‘Trixie’)' ],
	[ { os: 'macos' }, 'ONNX Runtime requires macOS ≥ 13.4' ],
	[ { os: 'ios' }, 'ONNX Runtime requires iOS ≥ 15.1' ],
	[ { os: 'android' }, 'ONNX Runtime requires Android ≥ 7.0 (API 24)' ]
];

function detectNativeOs(): Platform | null {
	const ua = navigator.userAgent;
	if (/windows|win64/i.test(ua)) {
		return 'windows';
	}
	if (/macintosh|macos/i.test(ua)) {
		return 'macos';
	}
	if (/android/i.test(ua)) {
		return 'android';
	}
	if (/iphone|ipad/i.test(ua)) {
		return 'ios';
	}
	if (/ubuntu|debian|gentoo|arch ?linux|fedora|centos|red ?hat|raspbian|deepin|manjaro|elementary ?os/i.test(ua)) {
		return 'linux';
	}
	return null;
}

export function detectNativeArch(platform: Platform | null): Arch | null {
	if (platform === 'macos') {
		try {
			const canvas = document.createElement('canvas');
			const gl = canvas.getContext('webgl');
			if (gl) {
				const extension = gl.getExtension('WEBGL_debug_renderer_info');
				if (extension && gl.getParameter(extension.UNMASKED_RENDERER_WEBGL)?.match(/apple m\d/i)) {
					return 'arm64';
				} else if (gl.getSupportedExtensions()?.indexOf('WEBGL_compressed_texture_etc') !== -1) {
					return 'arm64';
				}
			}
		} catch {}
	}

	if (platform === 'ios' || platform === 'android') {
		return 'arm64';
	}

	const ua = navigator.userAgent;
	if (/x86\b|i[3456]86\b/i.test(ua)) {
		return 'x86';
	} else if (/(amd|x)(64|86[-_]64)|wow64/i.test(ua)) {
		return 'x64';
	} else if (/aarch64|armv[89]|arm64/i.test(ua)) {
		return 'arm64';
	} else if (/armv[67]|armhf|armeabi/i.test(ua)) {
		return 'arm';
	}

	return null;
}

export const PLATFORM_STORE = create<{ [K in keyof Triple]: Triple[K] | null }>()(
	persist(
		() => {
			const os = detectNativeOs();
			const arch = detectNativeArch(os);
			return { os, arch };
		},
		{ name: 'platform', version: 0, storage: createJSONStorage(() => localStorage) }
	)
);
