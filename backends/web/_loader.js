const INIT_SYMBOL = Symbol('@ort-web.init');

const FEATURES_NONE = 0;
const FEATURES_WEBGL = 1 << 0;
const FEATURES_WEBGPU = 1 << 1;
const FEATURES_ALL = FEATURES_WEBGL | FEATURES_WEBGPU;

/**
 * @typedef {Object} Dist
 * @property {string} baseUrl
 * @property {string} scriptName
 * @property {string | null} [binaryName]
 * @property {string | null} [wrapperName] defaults to `binaryName` s/\.wasm$/.mjs
 * @property {Record<'main' | 'wrapper' | 'binary', string> | null} integrities
 */

const DEFAULT_DIST_BASE = 'https://cdn.pyke.io/0/pyke:ort-rs/web@1.24.3/';

/** @type {Record<number, Dist>} */
const DEFAULT_DIST = {
	[FEATURES_NONE]: {
		baseUrl: DEFAULT_DIST_BASE,
		scriptName: 'ort.wasm.min.js',
		binaryName: 'ort-wasm-simd-threaded.wasm',
		integrities: {
			main: '1SBQgvQsxJRGAOAJ6K2nPaLO1SKelZwoF+biXgv2/D9fPspYLhvG4WIMDb/BUoJC',
			wrapper: '/xM/eq8aUBJZgBuVwTQcLA5KlNmP6HOaENdJVgCkA/06cOMdL9EIQtmMuXOlMZEd',
			binary: 'sZw0EVBgUn+dNhQfjHDg8lwtmicKMm1bTvWS4rIRNxoVN1S9HkVyJ2nreMpYruEZ'
		}
	},
	[FEATURES_WEBGL]: {
		baseUrl: DEFAULT_DIST_BASE,
		scriptName: 'ort.webgl.min.js',
		binaryName: 'ort-wasm-simd-threaded.wasm',
		integrities: {
			main: 'tBplgcsp8KUmgpc7glC4gbb5MdN/LBa1x90w++Y5WHDLaLo0m75wK+OtxIVa0KB6',
			wrapper: '/xM/eq8aUBJZgBuVwTQcLA5KlNmP6HOaENdJVgCkA/06cOMdL9EIQtmMuXOlMZEd',
			binary: 'sZw0EVBgUn+dNhQfjHDg8lwtmicKMm1bTvWS4rIRNxoVN1S9HkVyJ2nreMpYruEZ'
		}
	},
	[FEATURES_WEBGPU]: {
		baseUrl: DEFAULT_DIST_BASE,
		scriptName: 'ort.webgpu.min.js',
		binaryName: 'ort-wasm-simd-threaded.jsep.wasm',
		integrities: {
			main: 'ryEl5/BLEvDIrNfBZGTpwZzs0EFe+rDt9wM/Xs5DbM7mwJm2V6/BPE7AGVtwKKiL',
			wrapper: 'C9DMcnZCIFFbpwJbX9QrGnrhpRt+2yD/FCcEZRrYf0iuzpOlCibPY0zsN8Wh1O+U',
			binary: 'SLG1FQY8ZmHhts4OFaia4WTuj6Ttjqb3U7uJQXb8L19O+HB0DG8zJh+vdmXdxD53'
		}
	},
	[FEATURES_ALL]: {
		baseUrl: DEFAULT_DIST_BASE,
		scriptName: 'ort.all.min.js',
		binaryName: 'ort-wasm-simd-threaded.jsep.wasm',
		integrities: {
			main: 'aSKRQamw4Hf1ggcSUR2ZFriC21oCir1YibdErfC+xmh/w1ijdn/GP4Vr4rMu1dvs',
			wrapper: 'C9DMcnZCIFFbpwJbX9QrGnrhpRt+2yD/FCcEZRrYf0iuzpOlCibPY0zsN8Wh1O+U',
			binary: 'SLG1FQY8ZmHhts4OFaia4WTuj6Ttjqb3U7uJQXb8L19O+HB0DG8zJh+vdmXdxD53'
		}
	}
};

/**
 * @param {string} url
 * @param {'fetch' | 'script' | 'module'} as
 * @param {string} [type]
 * @param {string | null} [integrity]
 */
function preload(url, as, type, integrity) {
	const el = document.createElement('link');
	el.href = url;
	if (as !== 'module') {
		el.rel = 'preload';
		el.setAttribute('as', as);
	} else {
		el.rel = 'modulepreload';
	}
	if (type) {
		el.setAttribute('type', type);
	}
	if (integrity) {
		el.setAttribute('integrity', `sha384-${integrity}`);
	}
	el.setAttribute('crossorigin', 'anonymous');
	document.head.appendChild(el);
}

/**
 * @param {number} features
 * @param {Dist} [dist]
 * @returns {Promise<boolean>}
 */
export function initRuntime(features, dist) {
	if ('ort' in window && /** @type {any} */(window).ort[INIT_SYMBOL]) {
		return Promise.resolve(false);
	}

	if (!dist) {
		if (!(features in DEFAULT_DIST)) {
			return Promise.reject(new Error('Unsupported feature set'));
		}

		dist = DEFAULT_DIST[features];
	}

	/** @param {string} file */
	const relative = file => new URL(file, dist.baseUrl).toString();

	return new Promise((resolve, reject) => {
		// since the order is load main script -> imports wrapper script -> fetches wasm, now would be a good time to
		// start fetching those
		if (dist.binaryName) {
			preload(
				relative(dist.binaryName),
				'fetch',
				'application/wasm',
				dist.integrities && dist.integrities.binary
			);
			preload(
				relative(dist.wrapperName || dist.binaryName.replace(/\.wasm$/, '.mjs')),
				'module',
				undefined,
				dist.integrities && dist.integrities.wrapper
			);
		}

		const script = document.createElement('script');
		script.src = new URL(dist.scriptName, dist.baseUrl).toString();
		if (dist.integrities && dist.integrities.main) {
			script.setAttribute('integrity', `sha384-${dist.integrities && dist.integrities.main}`);
		}
		script.setAttribute('crossorigin', 'anonymous');
		script.addEventListener('load', () => {
			if (!('ort' in window)) {
				return reject(new Error('script loaded but ort not defined'));
			}

			Object.defineProperty(window.ort, INIT_SYMBOL, {
				value: true,
				configurable: false,
				enumerable: false,
				writable: false
			});

			resolve(true);
		});
		script.addEventListener('error', e => {
			reject(e.error);
		});
		document.head.appendChild(script);
	});
}
