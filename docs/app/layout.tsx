import type { Metadata } from 'next';
import { Footer, Layout, Navbar, ThemeSwitch } from 'nextra-theme-docs';
import { Banner, Head, Search } from 'nextra/components';
import { getPageMap } from 'nextra/page-map';

import 'nextra-theme-docs/style.css';
import './globals.css';

export const metadata: Metadata = {
	metadataBase: new URL('https://ort.pyke.io'),
	title: {
		default: 'ort',
		template: '%s | ort'
	},
	description: 'ort is a Rust library for accelerated machine learning inference & training for ONNX models, providing bindings to Microsoft\'s ONNX Runtime and a wrapper around other Rust-native ML crates.',
	applicationName: 'ort Documentation'
};

export default async function RootLayout({ children }) {
	const pageMap = await getPageMap();
	return <html lang='en' dir='ltr' suppressHydrationWarning>
		<Head faviconGlyph='🦀'>

		</Head>
		<body>
			<Layout
				docsRepositoryBase='https://github.com/pykeio/ort/blob/main/docs'
				nextThemes={{
					defaultTheme: 'system'
				}}
				navbar={<Navbar
					logo={<img src="/assets/banner.png" style={{ height: '34px' }} />}
					chatLink='https://discord.gg/uQtsNu2xMa'
					projectLink='https://github.com/pykeio/ort'
				/>}
				footer={
					<Footer>
						<div>
							<p>made with 💜 by <a target="_blank" href="https://pyke.io/" style={{ textDecoration: 'underline', textDecorationColor: '#05c485' }}><span style={{ color: '#2ba9f6' }}>py</span><span style={{ color: '#00c875' }}>ke</span></a></p>
						</div>
					</Footer>
				}
				pageMap={pageMap}
			>
				{children}
			</Layout>
		</body>
	</html>
}
