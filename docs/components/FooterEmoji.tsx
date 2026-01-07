'use client';

import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';

const FOOTER_EMOJIS = [ '💜', '🩷', '💜', '🩷', '💜', '🩷', '💜', '🩷', '☕', '👽', '🦀', '🌈', '🏳️‍⚧️' ];

export default function FooterEmoji() {
	const route = usePathname();
	const [ emoji, setEmoji ] = useState(FOOTER_EMOJIS[0]);
	useEffect(() => {
		setEmoji(FOOTER_EMOJIS[Math.floor(Math.random() * FOOTER_EMOJIS.length)]);
	}, [ route ]);
	return emoji;
}
