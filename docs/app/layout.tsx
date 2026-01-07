import { Theme } from '@radix-ui/themes';
import type { Metadata } from 'next';
import { Layout, Navbar, Footer as NextraFooter } from 'nextra-theme-docs';
import { Head } from 'nextra/components';
import { getPageMap } from 'nextra/page-map';

import '@radix-ui/themes/styles.css';
import 'nextra-theme-docs/style.css';
import './globals.css';

import FooterEmoji from '../components/FooterEmoji';

export const metadata: Metadata = {
	metadataBase: new URL('https://ort.pyke.io'),
	title: {
		default: 'ort',
		template: '%s | ort'
	},
	description: 'ort is a Rust library for accelerated machine learning inference & training for ONNX models, providing bindings to Microsoft\'s ONNX Runtime and a wrapper around other Rust-native ML crates.',
	keywords: [
		'ort rust',
		'rust onnxruntime',
		'onnx runtime',
		'rust ai inference',
		'rust machine learning',
		'onnx rust'
	],
	applicationName: 'ort Documentation',
	appleWebApp: {
		capable: true,
		title: 'ort'
	},
	alternates: {
		canonical: './'
	},
	openGraph: {
		url: './',
		siteName: 'ort Docs',
		locale: 'en_US',
		type: 'website'
	},
	twitter: {
		site: 'https://ort.pyke.io'
	}
};

function Footer() {
	return <NextraFooter style={{ paddingTop: '18px', paddingBottom: '18px' }}>
		<div style={{ display: 'flex', flexDirection: 'column', width: '100%', alignItems: 'center', fontFamily: '"TASA Explorer"' }}>
			<span style={{ fontSize: '32px', fontWeight: '800', color: '#f74c00' }}>ort</span>
			<p style={{ fontWeight: '600' }}>
				made with <FooterEmoji /> by
				<a target="_blank" href="https://pyke.io/">
					<svg height='12' viewBox='0 0 21 10' style={{ display: 'inline', marginLeft: '5px', marginTop: '-4px' }}>
						<rect width='10' height='10' fill='#00BDFF' />
						<rect x='11' width='10' height='10' fill='#00FF86' />
					</svg>
				</a>
			</p>
		</div>
	</NextraFooter>
}

export default async function RootLayout({ children }) {
	const pageMap = await getPageMap();
	return <html lang='en' dir='ltr' suppressHydrationWarning>
		<Head faviconGlyph='🦀'>

		</Head>
		<body>
			<Theme accentColor="red" radius="large" scaling="105%">
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
					footer={<Footer />}
					pageMap={pageMap}
					copyPageButton={false}
				>
					{children}
				</Layout>
			</Theme>
		</body>
	</html>
}
