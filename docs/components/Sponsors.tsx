'use client';

import { useTheme } from 'nextra-theme-docs';
import { useEffect, useState } from 'react';

export function TocSponsors() {
	// cmon, i shouldnt have to do this
	const [ resolvedTheme, setTheme ] = useState('dark');
	const { resolvedTheme: actualResolvedTheme } = useTheme();
	useEffect(() => {
		setTheme(actualResolvedTheme ?? 'dark');
	}, [ actualResolvedTheme ]);

	return <>
		<div>
			<a href='https://rime.ai/' target="_blank">
				<img src={`https://cdn.pyke.io/0/pyke:ort-rs/docs@0.0.0/sponsor-identity/rime-${resolvedTheme}.svg`} alt='Rime.ai' suppressHydrationWarning />
			</a>
			<p style={{ color: 'var(--gray-11)', fontSize: '0.7rem' }}>Authentic AI voice models for enterprise.</p>
		</div>
	</>;
}
