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

const DIST_BASE = 'https://cdn.pyke.io/0/pyke:ort-rs/web@1.22.0/';

/** @type {Record<number, Dist>} */
const DIST = {
	[FEATURES_NONE]: {
		baseUrl: DIST_BASE,
		scriptName: 'ort.wasm.min.js',
		binaryName: 'ort-wasm-simd-threaded.wasm',
		integrities: {
			main: 'epp8GDQUoLKx5qHa6SoDHKv7fSHILsEYk4uEsHdPRBztXEIVWCe/lhhrQJUBMZcf',
			wrapper: 'LXMGGJ76ujT3yGw+OWQZVB6vBmJ7lqTO957Fh6ov3385aw3EncleBNFfYFAl3vXW',
			binary: 'Eu/XUdOA62yl+TueG792KtrQlAGAMW3g10sY4G3LBYyYZUtM126Z4Gr3ljTlXUGG'
		}
	},
	[FEATURES_WEBGL]: {
		baseUrl: DIST_BASE,
		scriptName: 'ort.webgl.min.js',
		binaryName: 'ort-wasm-simd-threaded.wasm',
		integrities: {
			main: 'IbmlOTVtLFqdmXae30hOMw60GXx+uyALrXF1TomZTqfkz2eL2RL/Po/TzbsGe/yv',
			wrapper: 'LXMGGJ76ujT3yGw+OWQZVB6vBmJ7lqTO957Fh6ov3385aw3EncleBNFfYFAl3vXW',
			binary: 'Eu/XUdOA62yl+TueG792KtrQlAGAMW3g10sY4G3LBYyYZUtM126Z4Gr3ljTlXUGG'
		}
	},
	[FEATURES_WEBGPU]: {
		baseUrl: DIST_BASE,
		scriptName: 'ort.webgpu.min.js',
		binaryName: 'ort-wasm-simd-threaded.jsep.wasm',
		integrities: {
			main: 'XM2cMlQFAUJFJ3s2424PSr/v9zkRT4aXfi1cUz2SunZatAOwTR5GfTcIKLJIf3Ns',
			wrapper: 'fZi+E4spXPUbkMSScLJlEGqj5QdfSJK7VQ2AZC5HLLV8lZg1j+TZT0RK6aEakeeX',
			binary: 'NNN1BawwGTHI+TPz2ivQSKo1AJHr/496DqG53T9IUQ6B9ruFNrov0DNJhqucIwZ1'
		}
	},
	[FEATURES_ALL]: {
		baseUrl: DIST_BASE,
		scriptName: 'ort.all.min.js',
		binaryName: 'ort-wasm-simd-threaded.jsep.wasm',
		integrities: {
			main: 'YWRTN6ucI4mQ8JMXfTaXD+iM7ExBj4KSHo6k6W9UIgx1tG98UgXpekjyYvRQ6akx',
			wrapper: 'fZi+E4spXPUbkMSScLJlEGqj5QdfSJK7VQ2AZC5HLLV8lZg1j+TZT0RK6aEakeeX',
			binary: 'NNN1BawwGTHI+TPz2ivQSKo1AJHr/496DqG53T9IUQ6B9ruFNrov0DNJhqucIwZ1'
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
		script.src = new URL(dist.binaryName, dist.baseUrl).toString();
		if (dist.integrities.main) {
			script.setAttribute('integrity', `sha384-${dist.integrities.main}`);
		}
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
