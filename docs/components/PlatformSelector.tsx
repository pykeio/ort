'use client';

import { Flex, Grid, Select, Text } from '@radix-ui/themes';
import { useCallback } from 'react';
import { PiAndroidLogoFill, PiAppleLogoFill, PiDeviceMobileCameraFill, PiGlobeBold, PiLinuxLogoFill, PiWindowsLogoFill } from 'react-icons/pi';
import { SiArm, SiIntel, SiWebassembly } from 'react-icons/si';
import { useStore } from 'zustand';

import { detectNativeArch, PLATFORM_STORE, type Arch, type Platform } from '../core/platform';
import styles from './PlatformSupport.module.css';

export default function PlatformSelector() {
	const selectedPlatform = useStore(PLATFORM_STORE);

	const onSelectOs = useCallback((os: Platform) => {
		if (os === 'web') {
			PLATFORM_STORE.setState({ os: 'web', arch: 'web' });
		} else {
			PLATFORM_STORE.setState(({ os: currentOs, arch: currentArch }) => {
				let arch = currentArch;
				if (currentOs === 'web') {
					arch = detectNativeArch(os);
				}
				if ((os === 'android' || os === 'ios') && currentArch !== 'arm') {
					arch = 'arm64';
				}
				return { os, arch };
			});
		}
	}, [ PLATFORM_STORE ]);
	const onSelectArch = useCallback((arch: Arch) => {
		PLATFORM_STORE.setState({ arch });
	}, [ PLATFORM_STORE ]);

	return <Grid columns={{ initial: '1', lg: '2' }} gap='3' width='100%'>
		<Flex direction='column' gap='1'>
			<Text size='2' weight='medium'>Operating system</Text>
			<Select.Root value={selectedPlatform.os ?? undefined} onValueChange={onSelectOs}>
				<Select.Trigger />
				<Select.Content position='popper'>
					<Select.Group>
						<Select.Item value={'windows' satisfies Platform} textValue='Windows'>
							<PiWindowsLogoFill className={styles.osIcon} />
							Windows
						</Select.Item>
						<Select.Item value={'macos' satisfies Platform} textValue='macOS'>
							<PiAppleLogoFill className={styles.osIcon} />
							macOS
						</Select.Item>
						<Select.Item value={'linux' satisfies Platform} textValue='Linux'>
							<PiLinuxLogoFill className={styles.osIcon} />
							Linux
						</Select.Item>
						<Select.Item value={'ios' satisfies Platform} textValue='iOS'>
							<PiDeviceMobileCameraFill className={styles.osIcon} />
							iOS
						</Select.Item>
						<Select.Item value={'android' satisfies Platform} textValue='Android'>
							<PiAndroidLogoFill className={styles.osIcon} />
							Android
						</Select.Item>
						<Select.Item value={'web' satisfies Platform} textValue='Web'>
							<PiGlobeBold className={styles.osIcon} />
							Web
						</Select.Item>
					</Select.Group>
				</Select.Content>
			</Select.Root>
		</Flex>
		<Flex direction='column' gap='1'>
			<Text size='2' weight='medium'>CPU architecture</Text>
			<Select.Root value={selectedPlatform.arch ?? undefined} onValueChange={onSelectArch} disabled={selectedPlatform.os === 'web'}>
				<Select.Trigger />
				<Select.Content position='popper'>
					<Select.Group>
						<Select.Item value={'x64' satisfies Arch} textValue='x86-64'>
							<SiIntel className={styles.osIcon} />
							x86-64
						</Select.Item>
						<Select.Item value={'arm64' satisfies Arch} textValue='ARM64'>
							<SiArm className={styles.osIcon} />
							ARM64
						</Select.Item>
						<Select.Item value={'x86' satisfies Arch} textValue='x86 (32-bit)' style={{ color: 'var(--gray-11)' }}>
							<SiIntel className={styles.osIcon} />
							x86 (32-bit)
						</Select.Item>
						<Select.Item value={'arm' satisfies Arch} textValue='ARMv7 (32-bit)' style={{ color: 'var(--gray-11)' }}>
							<SiArm className={styles.osIcon} />
							ARMv7 (32-bit)
						</Select.Item>
						<Select.Item value={'web' satisfies Arch} textValue='WASM' style={{ display: 'none' }}>
							<SiWebassembly className={styles.osIcon} />
							WASM
						</Select.Item>
					</Select.Group>
				</Select.Content>
			</Select.Root>
		</Flex>
	</Grid>;
}
