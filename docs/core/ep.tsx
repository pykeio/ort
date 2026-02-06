import { BiChip } from 'react-icons/bi';
import { BsAmd } from 'react-icons/bs';
import { PiGlobeBold } from 'react-icons/pi';
import { SiAndroid, SiApache, SiApple, SiHuawei, SiNvidia, SiQualcomm } from 'react-icons/si';

import type { TripleFilter } from './platform';

export interface ExecutionProvider {
	icon: React.ReactNode;
	vendor: string | null;
	name: string;
	feature: string;
	platforms: TripleFilter[];
	binaries?: TripleFilter[];
}

const INTEL_LOGO = <svg xmlns="http://www.w3.org/2000/svg" height="1em" viewBox="0 0 388 150"><path fill="#04C7FD" d="M0 2.1h28.1v28.1H0z"/><path fill="#0068B5" d="M27.4 148.5V47.3H.8v101.2h26.6zm176.8 1v-24.8c-3.9 0-7.2-.2-9.6-.6-2.8-.4-4.9-1.4-6.3-2.8a11.2 11.2 0 0 1-2.8-6c-.4-2.5-.6-5.8-.6-9.8V70.1h19.3V47.3h-19.3V7.8h-26.7v97.9a88 88 0 0 0 2.1 20.9c1.4 5.5 3.8 10 7.1 13.4s7.7 5.8 13 7.3a77.7 77.7 0 0 0 20.3 2.2h3.5zm152.8-1V0h-26.7v148.5H357zM132.5 57.2c-7.4-8-17.8-12-31-12a37.4 37.4 0 0 0-30.7 14.7l-1.5 1.9V47.3H43v101.2h26.5V94.6v3.7-1.8c.3-9.5 2.6-16.5 7-21a23 23 0 0 1 16.9-7.2c7.7 0 13.6 2.4 17.5 7a29.7 29.7 0 0 1 5.8 19.4v53.7h26.9V91a47.7 47.7 0 0 0-11.1-33.8zm184 40.5c0-7.3-1.3-14.1-3.8-20.5a48.7 48.7 0 0 0-27.2-27.9c-6.4-2.7-13.5-4-21.2-4a52.6 52.6 0 0 0-48.7 73.2 50 50 0 0 0 27.8 27.9 55.2 55.2 0 0 0 21.7 4.2 57.3 57.3 0 0 0 45-19.9l-19.2-14.6a35 35 0 0 1-25.6 11.3c-7.5 0-13.7-1.7-18.4-5.2a25.7 25.7 0 0 1-9.6-14.1l-.3-.9h79.5v-9.5zm-79.3-9.3c0-7.4 8.5-20.3 26.8-20.4 18.3 0 26.9 12.9 26.9 20.3l-53.7.1z"/></svg>;
const ARM_LOGO = <svg xmlns="http://www.w3.org/2000/svg" height="1em" viewBox="0 0 237 73"><path fill="#0091BD" d="M53.6 2.4h16v68h-16v-7c-7 8-15.5 9.2-20.4 9.2-21 0-33-17.5-33-36.2C.2 14.2 15.4.6 33.4.6c5 0 13.8 1.3 20.2 9.7v-8zM16.4 36.7c0 11.8 7.4 21.7 19 21.7 10 0 19.2-7.3 19.2-21.5 0-15-9.2-22-19.3-22-11.4 0-18.9 9.7-18.9 21.8zm72-34.3h15.8v6c1.8-2 4.4-4.3 6.6-5.6a18 18 0 0 1 9.7-2.3c4 0 8.1.6 12.5 3.2L126.7 18a14.1 14.1 0 0 0-8.1-2.4c-3.4 0-6.8.5-10 3.7-4.3 4.7-4.3 11.2-4.3 15.7v35.3h-16v-68zm54.8 0h16v6.3A20.8 20.8 0 0 1 176 .6a20 20 0 0 1 17.6 10 24 24 0 0 1 20.2-10c8.2 0 15.4 3.9 19.3 10.7 1.4 2.3 3.7 7.3 3.7 17.2v42h-16V33.2c0-7.6-.8-10.7-1.5-12.1a9 9 0 0 0-9-6c-4 0-7.4 2-9.5 5-2.8 3.9-3 9.7-3 15.5v35h-16V33.2c0-7.6-.8-10.7-1.5-12.1a9 9 0 0 0-9-6c-4 0-7.4 2-9.5 5-2.8 3.9-3 9.7-3 15.5v35h-15.6V2.5z"/></svg>;

export const EXECUTION_PROVIDER_ARRAY: ExecutionProvider[] = [
	{
		icon: <SiNvidia style={{ color: '#7bbb08' }} />,
		vendor: 'NVIDIA',
		name: 'CUDA',
		feature: 'cuda',
		platforms: [ { os: 'windows', arch: 'x64' }, { os: 'linux', arch: 'x64' }, { os: 'linux', arch: 'arm64' } ],
		binaries: [ { os: 'windows', arch: 'x64' }, { os: 'linux', arch: 'x64' } ]
	},
	{
		icon: <SiNvidia style={{ color: '#7bbb08' }} />,
		vendor: 'NVIDIA',
		name: 'TensorRT',
		feature: 'tensorrt',
		platforms: [ { os: 'windows', arch: 'x64' }, { os: 'linux', arch: 'x64' }, { os: 'linux', arch: 'arm64' } ],
		binaries: [ { os: 'windows', arch: 'x64' }, { os: 'linux', arch: 'x64' } ]
	},
	{
		icon: <SiNvidia style={{ color: '#7bbb08' }} />,
		vendor: 'NVIDIA',
		name: 'TensorRT RTX',
		feature: 'nvrtx',
		platforms: [ { os: 'windows', arch: 'x64' }, { os: 'linux', arch: 'x64' } ],
		binaries: [ { os: 'windows', arch: 'x64' }, { os: 'linux', arch: 'x64' } ]
	},
	{
		icon: <svg width="1em" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 22 22"><path fill="#f35325" d="M0 0h10v10H0z"/><path fill="#81bc06" d="M12 0h10v10H12z"/><path fill="#05a6f0" d="M0 12h10v10H0z"/><path fill="#ffba08" d="M12 12h10v10H12z"/></svg>,
		vendor: 'Microsoft',
		name: 'DirectML',
		feature: 'directml',
		platforms: [ { os: 'windows' } ],
		binaries: [ { os: 'windows' } ]
	},
	{
		icon: <SiApple />,
		vendor: 'Apple',
		name: 'CoreML',
		feature: 'coreml',
		platforms: [ { os: 'macos' }, { os: 'ios' } ],
		binaries: [ { os: 'macos', arch: 'arm64' }, { os: 'ios', arch: 'arm64' } ]
	},
	{
		icon: <BsAmd style={{ color: '#dd0823' }} />,
		vendor: 'AMD',
		name: 'MIGraphX',
		feature: 'migraphx',
		platforms: [ { os: 'windows', arch: 'x64' }, { os: 'linux', arch: 'x64' } ]
	},
	{
		icon: INTEL_LOGO,
		vendor: null,
		name: 'OpenVINO',
		feature: 'openvino',
		platforms: [ { os: 'windows', arch: 'x64' }, { os: 'linux', arch: 'x64' } ]
	},
	{
		icon: INTEL_LOGO,
		vendor: null,
		name: 'oneDNN',
		feature: 'onednn',
		platforms: [ { os: 'windows', arch: 'x64' }, { os: 'linux', arch: 'x64' } ]
	},
	{
		icon: <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="1em"><path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/><path d="M1 1h22v22H1z" fill="none"/></svg>,
		vendor: 'Google',
		name: 'XNNPACK',
		feature: 'xnnpack',
		platforms: [ { arch: 'x64' }, { arch: 'arm64' }, { arch: 'web' } ],
		binaries: [ { arch: 'x64' }, { arch: 'arm64' }, { arch: 'web' } ]
	},
	{
		icon: <SiQualcomm style={{ color: '#2e52dd' }} />,
		vendor: 'Qualcomm',
		name: 'QNN',
		feature: 'qnn',
		platforms: [ { os: 'windows', arch: 'arm64' }, { os: 'linux', arch: 'arm64' }, { os: 'android', arch: 'arm64' } ]
	},
	{
		icon: <SiHuawei style={{ color: '#d11233' }} />,
		vendor: 'Huawei',
		name: 'CANN',
		feature: 'cann',
		platforms: [ { os: 'linux', arch: 'arm64' }, { os: 'linux', arch: 'x64' } ]
	},
	{
		icon: <SiAndroid style={{ color: '#a7c83f' }} />,
		vendor: 'Android',
		name: 'NNAPI',
		feature: 'nnapi',
		platforms: [ { os: 'android' } ],
		binaries: [ { os: 'android' } ]
	},
	{
		icon: <SiApache style={{ color: '#a80808' }} />,
		vendor: 'Apache',
		name: 'TVM',
		feature: 'tvm',
		platforms: [{}]
	},
	{
		icon: ARM_LOGO,
		vendor: null,
		name: 'Arm Compute Library',
		feature: 'acl',
		platforms: [ { arch: 'arm64' } ]
	},
	{
		icon: <BsAmd style={{ color: '#dd0823' }} />,
		vendor: 'AMD',
		name: 'Vitis AI',
		feature: 'vitis',
		platforms: [ { os: 'windows', arch: 'x64' }, { os: 'linux', arch: 'x64' } ]
	},
	{
		icon: <BiChip />,
		vendor: 'Rockchip',
		name: 'RKNPU',
		feature: 'rknpu',
		platforms: [ { os: 'linux', arch: 'arm64' } ]
	},
	{
		icon: <svg xmlns="http://www.w3.org/2000/svg" width="1em" viewBox="0 0 96 96"><defs><linearGradient id="a" x1="-1032.172" x2="-1059.213" y1="145.312" y2="65.426" gradientTransform="matrix(1 0 0 -1 1075 158)" gradientUnits="userSpaceOnUse"><stop offset="0" stopColor="#114a8b"/><stop offset="1" stopColor="#0669bc"/></linearGradient><linearGradient id="b" x1="-1023.725" x2="-1029.98" y1="108.083" y2="105.968" gradientTransform="matrix(1 0 0 -1 1075 158)" gradientUnits="userSpaceOnUse"><stop offset="0" stopOpacity=".3"/><stop offset=".071" stopOpacity=".2"/><stop offset=".321" stopOpacity=".1"/><stop offset=".623" stopOpacity=".05"/><stop offset="1" stopOpacity="0"/></linearGradient><linearGradient id="c" x1="-1027.165" x2="-997.482" y1="147.642" y2="68.561" gradientTransform="matrix(1 0 0 -1 1075 158)" gradientUnits="userSpaceOnUse"><stop offset="0" stopColor="#3ccbf4"/><stop offset="1" stopColor="#2892df"/></linearGradient></defs><path fill="url(#a)" d="M33.338 6.544h26.038l-27.03 80.087a4.152 4.152 0 0 1-3.933 2.824H8.149a4.145 4.145 0 0 1-3.928-5.47L29.404 9.368a4.152 4.152 0 0 1 3.934-2.825z"/><path fill="#0078d4" d="M71.175 60.261h-41.29a1.911 1.911 0 0 0-1.305 3.309l26.532 24.764a4.171 4.171 0 0 0 2.846 1.121h23.38z"/><path fill="url(#b)" d="M33.338 6.544a4.118 4.118 0 0 0-3.943 2.879L4.252 83.917a4.14 4.14 0 0 0 3.908 5.538h20.787a4.443 4.443 0 0 0 3.41-2.9l5.014-14.777 17.91 16.705a4.237 4.237 0 0 0 2.666.972H81.24L71.024 60.261l-29.781.007L59.47 6.544z"/><path fill="url(#c)" d="M66.595 9.364a4.145 4.145 0 0 0-3.928-2.82H33.648a4.146 4.146 0 0 1 3.928 2.82l25.184 74.62a4.146 4.146 0 0 1-3.928 5.472h29.02a4.146 4.146 0 0 0 3.927-5.472z"/></svg>,
		vendor: 'Microsoft',
		name: 'Azure',
		feature: 'azure',
		platforms: [ { os: 'linux' }, { os: 'windows' }, { os: 'android' } ]
	},
	{
		icon: <svg xmlns="http://www.w3.org/2000/svg" height="1em" viewBox="0 0 768 600"><path fill="#0086e8" stroke="#0086e8" d="m626.63 295.5-60.189-104.25h120.55z"/><path fill="#0093ff" stroke="#0093ff" d="m626.63 87.001-60.189 104.25h120.55z"/><path fill="#0076cc" stroke="#0076cc" d="M506.26 504 385.88 295.5l240.76-.002z"/><path fill="#0066b0" stroke="#0066b0" d="M506.26 87 385.88 295.5l240.76-.002z"/><path fill="#005a9c" stroke="#005a9c" d="M265.5 504 24.74 87h481.51z"/></svg>,
		vendor: null,
		name: 'WebGPU',
		feature: 'webgpu',
		platforms: [ { os: 'web' }, { os: 'windows' }, { os: 'linux' } ],
		binaries: [ { os: 'web' }, { os: 'windows', arch: 'x64' }, { os: 'linux', arch: 'x64' } ]
	},
	{
		icon: <PiGlobeBold style={{ color: '#0066b0' }} />,
		vendor: null,
		name: 'WebGL',
		feature: 'webgl',
		platforms: [ { os: 'web' } ],
		binaries: [ { os: 'web' } ]
	},
	{
		icon: <PiGlobeBold style={{ color: '#0066b0' }} />,
		vendor: null,
		name: 'WebNN',
		feature: 'webnn',
		platforms: [ { os: 'web' } ],
		binaries: [ { os: 'web' } ]
	}
];
