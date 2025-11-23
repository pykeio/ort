import { FaRust } from 'react-icons/fa6';
import { PiAtomBold, PiBugBeetleFill, PiCubeBold, PiGearFill, PiHandCoinsFill, PiLightningFill, PiMapPinFill, PiMapTrifoldBold, PiWrenchFill } from 'react-icons/pi';
import { RxArrowTopRight } from 'react-icons/rx';

import { CRATE_VERSION } from '../constants';

export default {
	"link-oc": {
		title: <><PiHandCoinsFill className='sidebar-icon' /> Sponsor <RxArrowTopRight /></>,
		href: "https://opencollective.com/pyke-osai",
	},
	"link-api": {
		title: <><FaRust className='sidebar-icon' /> API Reference <RxArrowTopRight /></>,
		href: `https://docs.rs/ort/${CRATE_VERSION}/ort/`
	},
	"link-crates": {
		title: <><PiCubeBold className='sidebar-icon' /> Crates.io <RxArrowTopRight /></>,
		href: `https://crates.io/crates/ort/${CRATE_VERSION}`,
	},
	"-- Docs": {
		"type": "separator",
		"title": "Docs"
	},
	"index": <b><PiMapPinFill className='sidebar-icon-small' style={{ marginRight: '10px' }} /> Introduction</b>,
	"setup": {
		"title": <b><PiWrenchFill className='sidebar-icon-small' style={{ marginRight: '10px' }} /> Setup</b>
	},
	"fundamentals": {
		"title": <b><PiAtomBold className='sidebar-icon-small' style={{ marginRight: '10px' }} /> Fundamentals</b>
	},
	"perf": {
		"title": <b><PiLightningFill className='sidebar-icon-small' style={{ marginRight: '10px' }} /> Performance</b>
	},
	"backends": {
		"title": <b><PiGearFill className='sidebar-icon-small' style={{ marginRight: '10px' }} /> Backends</b>
	},
	"troubleshooting": {
		"title": <b><PiBugBeetleFill className='sidebar-icon-small' style={{ marginRight: '10px' }} /> Troubleshooting</b>
	},
	"migrating": {
		"title": <b><PiMapTrifoldBold className='sidebar-icon-small' style={{ marginRight: '10px' }} /> Migration & versioning</b>
	}
};
