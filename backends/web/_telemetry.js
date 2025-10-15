const EVENT_URL = 'https://signal.pyke.io/beacon/9f5be487-d137-455a-9938-2fc7ecaa9de3/vVOv73JqP3iYRqXMBNm';

const IS_LOCALHOST = /^localhost$|^127(\.[0-9]+){0,2}\.[0-9]+$|^\[::1?\]$/;

/** @param {Uint8Array<ArrayBuffer>} payload */
function track(payload) {
	if (IS_LOCALHOST.test(location.hostname) || location.protocol === 'file:') {
		return false;
	}
	if (navigator.webdriver || 'Cypress' in window) {
		return false;
    }

	return navigator.sendBeacon(EVENT_URL, payload.buffer);
}

/** @param {Uint8Array<ArrayBuffer>[]} chunks */
function concat(...chunks) {
	const concatenated = new Uint8Array(chunks.reduce((a, b) => a + b.byteLength, 0));
	let offset = 0;
	for (const chunk of chunks) {
		concatenated.set(chunk, offset);
		offset += chunk.byteLength;
	}
	return concatenated;
}

/** @param {number} x */
function asUint32(x) {
	const view = new DataView(new ArrayBuffer(4));
	view.setUint32(0, x, true);
	return new Uint8Array(view.buffer);
}

const encoder = new TextEncoder();

let hasInitializedSession = false;
export function trackSessionInit() {
	if (hasInitializedSession) {
		return true;
	}

	hasInitializedSession = true;

	const hostname = location.hostname;
	return track(concat(
		new Uint8Array([ 0x01 ]),
		new Uint8Array([ 0x90, 0x63, 0x8A, 0xE7 ]),
		asUint32(hostname.length),
		encoder.encode(hostname)
	));
}
