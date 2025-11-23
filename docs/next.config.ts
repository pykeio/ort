import nextra from 'nextra';

export default nextra({
	search: {
		codeblocks: true
	},
	codeHighlight: true,
	defaultShowCopyCode: true,
	contentDirBasePath: '/'
})({
	reactStrictMode: true,
	output: 'export',
	images: {
		unoptimized: true
	}
});
