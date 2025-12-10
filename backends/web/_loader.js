const INIT_SYMBOL = Symbol('@ort-web.init');

const FEATURES_NONE = 0;
const FEATURES_WEBGL = 1 << 0;
const FEATURES_WEBGPU = 1 << 1;
const FEATURES_ALL = FEATURES_WEBGL | FEATURES_WEBGPU;

/**
 * @typedef {Object} Dist
 * @property {string} baseUrl
 * @property {string} scriptName
 * @property {string} binaryName
 * @property {string} [wrapperName] defaults to `binaryName` s/\.wasm$/.mjs
 * @property {Record<'main' | 'wrapper' | 'binary', string>} integrities
 */

const DIST_BASE = 'https://cdn.pyke.io/0/pyke:ort-rs/web@1.23.0/';

/** @type {Record<number, Dist>} */
const DIST = {
	[FEATURES_NONE]: {
		baseUrl: DIST_BASE,
		scriptName: 'ort.wasm.min.js',
		binaryName: 'ort-wasm-simd-threaded.wasm',
		integrities: {
			main: 'Uvpo3KshAzID7bmsY+Pz2/tiNWwl6Y5XeDTPpktDx73e0o/1TdssZDScTVHxpLYv',
			wrapper: 'Y/ZaWdP4FERyRvi+anEVDVDDhMJKldzf33TRb2MiCALo054swqCUe6aM/tD8XL6g',
			binary: '9UMXJFWi2zyn9PbGgXmJjEYM4hu8T8zmqmgxX6zQ08ZmNBOso3IT0cTp3M3oU7DU'
		}
	},
	[FEATURES_WEBGL]: {
		baseUrl: DIST_BASE,
		scriptName: 'ort.webgl.min.js',
		binaryName: 'ort-wasm-simd-threaded.wasm',
		integrities: {
			main: 'pD9jsAlDhP5yhHaVikKM6mXw/E4HPB+4kc/rf3lrMctGWwT0XpIxiTdH/XDHR7Pr',
			wrapper: 'Y/ZaWdP4FERyRvi+anEVDVDDhMJKldzf33TRb2MiCALo054swqCUe6aM/tD8XL6g',
			binary: '9UMXJFWi2zyn9PbGgXmJjEYM4hu8T8zmqmgxX6zQ08ZmNBOso3IT0cTp3M3oU7DU'
		}
	},
	[FEATURES_WEBGPU]: {
		baseUrl: DIST_BASE,
		scriptName: 'ort.webgpu.min.js',
		binaryName: 'ort-wasm-simd-threaded.jsep.wasm',
		integrities: {
			main: 'rY/SpyGuo298HuKPNCTIhlm3xc022++95XwJnuGVpKaW4yEzMTTDvgXoRQdiicvj',
			wrapper: 'Liv6LVoHkWBuJEPAGGmpzPGesXdc9YN5Eu0UaA9a9qChwB0H21V86UFBLhnIBieb',
			binary: 'jVPVL8reOtRz4+v3ZZAWg8bO5m7HGJr7tsMxmvNae28TztYbHZIk8JXHeZ/82yST'
		}
	},
	[FEATURES_ALL]: {
		baseUrl: DIST_BASE,
		scriptName: 'ort.all.min.js',
		binaryName: 'ort-wasm-simd-threaded.jsep.wasm',
		integrities: {
			main: 'VVNyVdgdgHOM/8agRDy7rVx66N+/9T1vkYzwYtSS/u36YVzaln3cMtxt24ozySvr',
			wrapper: 'Liv6LVoHkWBuJEPAGGmpzPGesXdc9YN5Eu0UaA9a9qChwB0H21V86UFBLhnIBieb',
			binary: 'jVPVL8reOtRz4+v3ZZAWg8bO5m7HGJr7tsMxmvNae28TztYbHZIk8JXHeZ/82yST'
		}
	}
};

/**
 * @param {string} url
 * @param {'fetch' | 'script' | 'module'} as
 * @param {string} [type]
 * @param {string} [integrity]
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
 * @returns {Promise<boolean>}
 */
export function initRuntime(features) {
	if ('ort' in window && /** @type {any} */(window).ort[INIT_SYMBOL]) {
		return Promise.resolve(false);
	}

	if (!(features in DIST)) {
		return Promise.reject(new Error('Unsupported feature set'));
	}

	const dist = DIST[features];
	/** @param {string} file */
	const relative = file => new URL(file, dist.baseUrl).toString();

	return new Promise((resolve, reject) => {
		// since the order is load main script -> imports wrapper script -> fetches wasm, now would be a good time to
		// start fetching those
		preload(
			relative(dist.binaryName),
			'fetch',
			'application/wasm',
			dist.integrities.binary
		);
		preload(
			relative(dist.wrapperName || dist.binaryName.replace(/\.wasm$/, '.mjs')),
			'module',
			undefined,
			dist.integrities.wrapper
		);

		const script = document.createElement('script');
		script.src = new URL(dist.scriptName, dist.baseUrl).toString();
		if (dist.integrities.main) {
			script.setAttribute('integrity', `sha384-${dist.integrities.main}`);
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
