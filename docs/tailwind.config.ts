import type { Config } from 'tailwindcss';

export default {
	content: [
		'./pages/**/*.{js,ts,jsx,tsx,md,mdx}'
	],
	darkMode: 'selector'
} as const satisfies Config;
