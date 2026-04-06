'use client';

import { Card } from '@radix-ui/themes';
import { useTheme } from 'nextra-theme-docs';
import { useEffect, useState } from 'react';

export function TocSponsors() {
	// cmon, i shouldnt have to do this
	const [ resolvedTheme, setTheme ] = useState('dark');
	const { resolvedTheme: actualResolvedTheme } = useTheme();
	useEffect(() => {
		setTheme(actualResolvedTheme ?? 'dark');
	}, [ actualResolvedTheme ]);

	return null;

	return <Card>
		<p style={{ fontFamily: '"Monaspace Neon"', textTransform: 'uppercase', color: 'var(--gray-9)', fontSize: '0.6rem', marginTop: '-4px', marginBottom: '4px' }}>Sponsored by</p>
		{/* ... */}
	</Card>;
}
