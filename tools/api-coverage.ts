import { dirname, join } from 'jsr:@std/path@1.0.3';
import { walk } from 'jsr:@std/fs@1.0.2';

const PROJECT_ROOT = dirname(import.meta.dirname!);
const DECODER = new TextDecoder('utf-8');
const SYMBOL_DEF_REGEX = /pub\s+([A-Za-z_][A-Za-z0-9_]+):/;
const SYMBOL_USAGE_REGEX = /ortsys!\[\s*(?:unsafe\s+)?([A-Za-z_][A-Za-z0-9_]+)/gm;

const IGNORED_SYMBOLS = new Set<string>([
	'CreateEnv', // we will always create an env with a custom logger for integration w/ tracing
]);

const sysSymbols = new Set<string>();
const sysFile = await Deno.readFile(join(PROJECT_ROOT, 'ort-sys', 'src', 'lib.rs'));
let isInOrtApi = false;
for (const line of DECODER.decode(sysFile).split('\n')) {
	if (line === 'pub struct OrtApi {') {
		isInOrtApi = true;
		continue;
	}

	if (isInOrtApi) {
		if (line === '}') {
			isInOrtApi = false;
			continue;
		}

		const trimmedLine = line.trimStart();
		if (SYMBOL_DEF_REGEX.test(trimmedLine)) {
			const [ _, symbol ] = trimmedLine.match(SYMBOL_DEF_REGEX)!;
			sysSymbols.add(symbol);
		}
	}
}

const usedSymbols = new Set<string>();
for await (const sourceFile of walk(join(PROJECT_ROOT, 'src'))) {
	if (sourceFile.isDirectory) {
		continue;
	}

	const contents = DECODER.decode(await Deno.readFile(sourceFile.path));
	for (const [ _, symbol ] of contents.matchAll(SYMBOL_USAGE_REGEX)) {
		usedSymbols.add(symbol);
	}
}

const unusedSymbols = sysSymbols
	.difference(usedSymbols)
	.difference(IGNORED_SYMBOLS);
for (const symbol of unusedSymbols) {
	console.log(`%c\t${symbol}`, 'color: red');
}
console.log(`%cCoverage: ${usedSymbols.size}/${sysSymbols.size} (${((usedSymbols.size / sysSymbols.size) * 100).toFixed(2)}%)`, 'color: green; font-weight: bold');
