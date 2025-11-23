'use client';

import { Flex, Skeleton, Text } from '@radix-ui/themes';
import confetti from 'canvas-confetti';
import { Link } from 'nextra-theme-docs';
import { Callout, Code } from 'nextra/components';
import { useEffect, useMemo } from 'react';
import { PiInfoBold, } from 'react-icons/pi';
import { useStore } from 'zustand/react';

import { PLATFORM_NOTES, PLATFORM_STORE, PLATFORMS_SUPPORTED, PLATFORMS_WITH_BINARIES } from '../core/platform';
import { useIsClient } from '../core/utils';
import Ort from './Ort';
import PlatformSelector from './PlatformSelector';
import styles from './PlatformSupport.module.css';

type SupportState = 'TIER1' | 'TIER2' | 'UNSUPPORTED';

const SUPPORT_STATES: Record<SupportState, { [K in 'emoji' | 'heading' | 'detail']: React.ReactNode }> = {
	TIER1: {
		emoji: '🎉',
		heading: 'Good news! You can dive right in!',
		detail: <><Ort /> provides binaries for this setup. Just add it to your <Code>Cargo.toml</Code> to get started.</>
	},
	TIER2: {
		emoji: '✅',
		heading: 'This setup is supported',
		detail: <><Ort /> doesn't have precompiled binaries for it, though, so you'll have to compile ONNX Runtime from source.</>
	},
	UNSUPPORTED: {
		emoji: '😕',
		heading: 'Sorry, that\'s a no',
		detail: <>ONNX Runtime doesn't support this setup. An <Link href='/backends'>alternative backend</Link> might, though!</>
	}
};

export default function PlatformSupport() {
	const isClient = useIsClient();
	const selectedPlatform = useStore(PLATFORM_STORE);

	const hasBinaries = useMemo(() => {
		if (!isClient) {
			return false;
		}

		const { os, arch } = selectedPlatform;
		for (const x of PLATFORMS_WITH_BINARIES) {
			if (x.arch === arch && x.os === os) {
				return true;
			}
		}
		return false;
	}, [ isClient, selectedPlatform.os, selectedPlatform.arch ]);
	const isSupported = useMemo(() => {
		if (!isClient) {
			return false;
		}

		const { os, arch } = selectedPlatform;
		for (const x of PLATFORMS_SUPPORTED) {
			if (x.arch === arch && x.os === os) {
				return true;
			}
		}
		return false;
	}, [ isClient, selectedPlatform.os, selectedPlatform.arch ]);
	const notes = useMemo(() => {
		const notes: React.ReactNode[] = [];
		const { os, arch } = selectedPlatform;
		for (const [ filter, note ] of PLATFORM_NOTES) {
			let applicable = true;
			if (filter.os !== undefined) {
				applicable &&= filter.os === os;
			}
			if (filter.arch !== undefined) {
				applicable &&= filter.arch === arch;
			}

			if (applicable) {
				notes.push(note);
			}
		}
		return notes;
	}, [ isClient, selectedPlatform.os, selectedPlatform.arch ]);

	useEffect(() => {
		if (isClient && hasBinaries) {
			confetti({
				particleCount: 100,
				angle: 60,
				spread: 55,
				origin: { x: 0 },
				ticks: 100,
				disableForReducedMotion: true
			});
			confetti({
				particleCount: 100,
				angle: 120,
				spread: 55,
				origin: { x: 1 },
				ticks: 100,
				disableForReducedMotion: true
			});
		}
	}, [ isClient, selectedPlatform ]);

	if (!isClient) {
		return <Skeleton width='100%' height='300px' my='2' />;
	}

	const stateKey: SupportState = hasBinaries
		? 'TIER1'
		: isSupported
			? 'TIER2'
			: 'UNSUPPORTED';
	const state = SUPPORT_STATES[stateKey];

	return <Flex direction='column' align='center' width='100%' minHeight='300px' my='2'>
		<PlatformSelector />
		<Flex direction='column' align={{ lg: 'center' }} my='9' gap='6' width='fit-content'>
			<Flex direction={{ initial: 'column', lg: 'row' }} align='center' justify={{ initial: 'center', lg: 'start' }} width='fit-content' gap='3'>
				<Text size='9'>{state.emoji}</Text>
				<Flex direction='column' gap='1' maxWidth='430px'>
					<Text size='5' weight='bold'>{state.heading}</Text>
					<Text>
						{selectedPlatform.os !== 'web'
							? state.detail
							: <>See the <Link href='/backends/web'><Code>ort-web</Code></Link> backend to setup <Ort /> for the Web.</>}
					</Text>
				</Flex>
			</Flex>
			{notes.length > 0 && <Flex direction='column' style={{ alignSelf: 'stretch' }} gap='2' className={styles.noteContainer}>
				<Text color='gray' weight='medium'><PiInfoBold className={styles.osIcon} />You should know...</Text>
				{notes.map((note, i) => <Callout key={i} type='default'>
					{note}
				</Callout>)}
			</Flex>}
		</Flex>
	</Flex>;
}
