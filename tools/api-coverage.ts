import { dirname, join } from 'jsr:@std/path@1.0.3';
import { walk } from 'jsr:@std/fs@1.0.2';

const PROJECT_ROOT = dirname(import.meta.dirname!);
const DECODER = new TextDecoder('utf-8');
const SYMBOL_DEF_REGEX = /pub\s+([A-Za-z_][A-Za-z0-9_]+):/;
const SYMBOL_USAGE_REGEX = /ortsys!\[\s*(?:unsafe\s+)?([A-Za-z_][A-Za-z0-9_]+)/gm;

const IGNORED_SYMBOLS = new Set<string>([
	'KernelContext_GetScratchBuffer', // implemented in src/operator/kernel.rs but impl appears to be broken so ignoring
	'RegisterCustomOpsLibrary', // we use RegisterCustomOpsLibrary_V2
	'RegisterCustomOpsUsingFunction',
	'SessionOptionsAppendExecutionProvider_CUDA', // we use V2
	'SessionOptionsAppendExecutionProvider_TensorRT', // we use V2
	'GetValueType', // we get value types via GetTypeInfo -> GetOnnxTypeFromTypeInfo, which is equivalent
	'SetLanguageProjection', // someday we shall have `ORT_PROJECTION_RUST`, but alas, today is not that day...

	// we use allocator APIs directly on the Allocator struct
	'AllocatorAlloc',
	'AllocatorFree',
	'AllocatorGetInfo',

	// functions that don't make sense with SessionBuilder API
	'HasSessionConfigEntry',
	'GetSessionConfigEntry',
	'DisableProfiling',
	'GetCUDAProviderOptionsAsString',
	'GetTensorRTProviderOptionsAsString',
	'GetCANNProviderOptionsAsString',
	'GetDnnlProviderOptionsAsString',
	'GetROCMProviderOptionsAsString',
	'GetTensorRTProviderOptionsByName',
	'GetCUDAProviderOptionsByName',

	// maybe these are meant to be used for custom ops?
	'CreateOpaqueValue',
	'GetOpaqueValue',

	// non-use
	'HasValue',
	'GetExecutionProviderApi', // available via ort_sys for those who need it
	'CreateCpuMemoryInfo',
	'ReleaseMapTypeInfo', // neither map or sequence type infos ever get directly allocated, so im not sure why these exist
	'ReleaseSequenceTypeInfo',
	'UpdateTensorRTProviderOptionsWithValue',
	'UpdateCUDAProviderOptionsWithValue'
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

const nonIgnoredSymbols = sysSymbols.difference(IGNORED_SYMBOLS);
const unusedSymbols = nonIgnoredSymbols.difference(usedSymbols);
for (const symbol of unusedSymbols) {
	console.log(`%c\t${symbol}`, 'color: red');
}
console.log(`%cCoverage: ${usedSymbols.size}/${nonIgnoredSymbols.size} (${((usedSymbols.size / nonIgnoredSymbols.size) * 100).toFixed(2)}%)`, 'color: green; font-weight: bold');
