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
	</>;
}
