import { createHash } from 'crypto';
import { readFile } from 'fs/promises';
import { join } from 'path';

import { ImageResponse } from 'next/og';
import type { MdxFile, PageMapItem } from 'nextra';
import { getPageMap } from 'nextra/page-map';

import meta from '../../../content/_meta';

function *flattenPageMap(items: PageMapItem[]): Generator<MdxFile> {
	for (const item of items) {
		if ('children' in item) {
			for (const child of item.children) {
				yield *flattenPageMap([ child ]);
			}
		} else if ('route' in item) {
			yield item;
		}
	}
}

const hash = (x: string) => createHash('sha256').update(x).digest('base64url');

export async function generateStaticParams() {
	return [ ...flattenPageMap(await getPageMap()).map(x => ({ id: hash(x.route) + '.png' })) ];
}

export async function GET(request: Request, { params }: { params: Promise<{ id: string }> }) {
	let { id } = await params;
	id = id.replace(/\.png$/, '');

	const file = flattenPageMap(await getPageMap()).find(x => hash(x.route) === id);
	if (!file) {
		throw new Error('unknown id');
	}

	const categorySlash = file.route.slice(1).indexOf('/');
	const categoryId = categorySlash !== -1 ? file.route.slice(1, 1 + categorySlash) : null;
	const categoryName = categoryId ? meta[categoryId]?.title : null;

	const explorer = await readFile(join(process.cwd(), 'app/og/[id]/TASAExplorer-Bold.otf'));
	const orbiter = await readFile(join(process.cwd(), 'app/og/[id]/TASAOrbiter-Regular.otf'));
	const neon = await readFile(join(process.cwd(), 'app/og/[id]/MonaspaceNeon-Regular.otf'));

	return new ImageResponse(
		(
			<div style={{
				backgroundImage: `url("https://ort.pyke.io/_og_template.png")`,
				width: '100%',
				height: '100%',
				padding: '70px',
				display: 'flex',
				flexDirection: 'column',
				justifyContent: 'flex-end',
				color: '#fff',
				gap: '12px'
			}}>
				<h3 style={{ lineHeight: '1', margin: '0', fontFamily: '"TASA Explorer"', fontSize: '30px', color: '#f74c00', textShadow: '0 0 3px #f74c00A0' }}>{categoryName}</h3>
				<h1 style={{ lineHeight: '0.9', margin: '0', fontFamily: '"TASA Explorer"', fontSize: '72px', maxWidth: '700px' }}>{file.frontMatter?.title ?? (file as any).title ?? file.name}</h1>
				{file.frontMatter?.description
					&& <p style={{
						opacity: '0.7',
						margin: '0',
						fontFamily: '"TASA Orbiter"',
						maxWidth: '740px',
						maxHeight: '240px',
						overflow: 'hidden',
						fontSize: '18px'
					}}>
						{file.frontMatter?.description}
					</p>}
			</div>
		),
		{
			width: 1200,
			height: 630,
			fonts: [
				{
					name: 'TASA Explorer',
					data: explorer
				},
				{
					name: 'TASA Orbiter',
					data: orbiter
				},
				{
					name: 'Monaspace Neon',
					data: neon
				}
			]
		}
	)
}
