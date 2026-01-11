import { createHash } from 'crypto';
import type { Metadata } from 'next';
import { generateStaticParamsFor, importPage } from 'nextra/pages';
import { useMDXComponents as getMDXComponents } from '../../mdx-components';

const hash = (x: string) => createHash('sha256').update(x).digest('base64url');

export const generateStaticParams = generateStaticParamsFor('mdxPath');

export async function generateMetadata(props): Promise<Metadata> {
	const params = await props.params;
	const { metadata } = await importPage(params.mdxPath);
	return {
		title: metadata.title,
		description: metadata.description,
		openGraph: {
			type: 'website',
			siteName: 'ort Docs',
			title: metadata.title,
			description: metadata.description ?? undefined,
			locale: 'en_US',
			images: [
				{
					url: `https://ort.pyke.io/og/${hash(`/${params.mdxPath?.join('/') ?? ''}`)}.png`,
					width: 1200,
					height: 630,
					type: 'image/png'
				}
			]
		},
		twitter: {
			card: 'summary_large_image',
			title: metadata.title,
			description: metadata.description ?? undefined,
			images: [
				{
					url: `https://ort.pyke.io/og/${hash(`/${params.mdxPath?.join('/') ?? ''}`)}.png`,
					width: 1200,
					height: 630,
					type: 'image/png'
				}
			]
		}
	};
}

const Wrapper = getMDXComponents().wrapper;

export default async function DocsPage(props) {
	const params = await props.params;
	const result = await importPage(params.mdxPath);
	const { default: MDXContent, toc, metadata, sourceCode } = result;
	return <Wrapper toc={toc} metadata={metadata} sourceCode={sourceCode}>
		<MDXContent {...props} params={params} />
	</Wrapper>
}
