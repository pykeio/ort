import nextra from 'nextra';

export default nextra({
	theme: 'nextra-theme-docs',
	themeConfig: './theme.config.jsx'
})({
	output: 'export',
	images: {
		unoptimized: true
	}
});
