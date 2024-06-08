import Image from 'next/image';

/** @type {import('nextra-theme-docs').DocsThemeConfig} */
const config = {
	project: {
		link: 'https://github.com/pykeio/ort'
	},
	chat: {
		link: 'https://discord.gg/uQtsNu2xMa'
	},
	docsRepositoryBase: 'https://github.com/pykeio/ort/blob/main/docs',
	useNextSeoProps() {
		return {
			titleTemplate: '%s | ort'
		}
	},
	logo: <img src="/assets/banner.png" style={{ height: '34px' }} />,
	darkMode: true,
	nextThemes: {
		defaultTheme: 'system'
	},
	footer: {
		text: <div>
			<p>made with 💜 by <a target="_blank" href="https://pyke.io/" style={{ textDecoration: 'underline', textDecorationColor: '#05c485' }}><span style={{ color: '#2ba9f6' }}>py</span><span style={{ color: '#00c875' }}>ke</span></a> • <a target="_blank" href="https://opencollective.com/pyke-osai" style={{ textDecoration: 'underline' }}>sponsor</a></p>
		</div>
	},
	primaryHue: 20,
	primarySaturation: 100,
	toc: {
		float: true
	}
};
export default config;
