'use client';

import { Card, Flex, Heading, Skeleton, Switch, Text } from '@radix-ui/themes';
import { useState } from 'react';
import { useStore } from 'zustand';

import { EXECUTION_PROVIDER_ARRAY } from '../core/ep';
import { PLATFORM_STORE } from '../core/platform';
import { useIsClient } from '../core/utils';
import PlatformSelector from './PlatformSelector';
import { Code } from 'nextra/components';
import { PiCheckBold, PiInfoFill, PiWarningBold, PiWarningFill } from 'react-icons/pi';

export default function ExecutionProviders() {
	const isClient = useIsClient();
	const selectedPlatform = useStore(PLATFORM_STORE);
	const [ showUnsupported, setShowUnsupported ] = useState(false);

	if (!isClient) {
		return <Skeleton width='100%' height='300px' my='4' />;
	}

	return <Flex direction='column' width='100%' minHeight='300px' my='4' gap='1'>
		<PlatformSelector />
		<Text as='label' size='3'>
			<Flex gap='2'>
				<Switch size='1' checked={showUnsupported} onCheckedChange={setShowUnsupported} />
				Show unsupported EPs
			</Flex>
		</Text>
		<Flex direction='column' gap='2'>
			{EXECUTION_PROVIDER_ARRAY
				.map(ep => {
					let supported = false;
					for (const filter of ep.platforms) {
						let applicable = true;
						if (filter.os !== undefined) {
							applicable &&= filter.os === selectedPlatform.os;
						}
						if (filter.arch !== undefined) {
							applicable &&= filter.arch === selectedPlatform.arch;
						}

						supported ||= applicable;
					}

					return { ...ep, supported };
				})
				.filter(ep => {
					if (!showUnsupported) {
						return ep.supported;
					} else {
						return true;
					}
				})
				.map(ep => {
					let hasBinaries = false;
					for (const filter of ep.binaries ?? []) {
						let applicable = true;
						if (filter.os !== undefined) {
							applicable &&= filter.os === selectedPlatform.os;
						}
						if (filter.arch !== undefined) {
							applicable &&= filter.arch === selectedPlatform.arch;
						}

						hasBinaries ||= applicable;
					}

					return { ...ep, hasBinaries };
				})
				.sort((a, b) => a.hasBinaries < b.hasBinaries ? 1 : -1)
				.map(ep => {
					return <Card key={ep.feature} style={!ep.supported ? { opacity: '0.6' } : undefined}>
						{!ep.supported && <Flex direction='row' gap='1' align='center' style={{ fontSize: '12px', color: 'var(--gray-10)' }}><PiWarningFill /> Unsupported</Flex>}
						<Flex direction='row' gap='1' align='center'>{ep.icon} {ep.vendor && <Text weight='medium'>{ep.vendor}</Text>}</Flex>
						<Heading size='5'>{ep.name}</Heading>
						{ep.supported
							? ep.hasBinaries
								? <Flex direction='row' gap='1' align='center' style={{ fontSize: '12px', color: 'var(--green-11)' }}><PiCheckBold /> Ready to use</Flex>
								: <Flex direction='row' gap='1' align='center' style={{ fontSize: '12px', color: 'var(--gray-11)' }}><PiInfoFill /> Requires compiling ONNX Runtime from source</Flex>
							: null}
						<Code style={{ fontSize: '12px' }}>features = [ "{ep.feature}" ]</Code>
					</Card>
				})}
		</Flex>
	</Flex>
}
