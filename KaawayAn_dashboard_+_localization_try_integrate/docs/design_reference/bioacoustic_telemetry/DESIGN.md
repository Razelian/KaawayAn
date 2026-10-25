---
name: Bioacoustic Telemetry
colors:
  surface: '#effdf4'
  surface-dim: '#cfddd5'
  surface-bright: '#effdf4'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#e9f7ef'
  surface-container: '#e3f1e9'
  surface-container-high: '#ddebe3'
  surface-container-highest: '#d8e6de'
  on-surface: '#121e19'
  on-surface-variant: '#414943'
  inverse-surface: '#27332e'
  inverse-on-surface: '#e6f4ec'
  outline: '#717973'
  outline-variant: '#c1c8c2'
  surface-tint: '#EEF6ED'
  primary: '#063423'
  on-primary: '#ffffff'
  primary-container: '#214b38'
  on-primary-container: '#8ebaa2'
  inverse-primary: '#a3d1b7'
  secondary: '#755843'
  on-secondary: '#ffffff'
  secondary-container: '#ffd8bd'
  on-secondary-container: '#7a5d47'
  tertiary: '#272f29'
  on-tertiary: '#ffffff'
  tertiary-container: '#3d453f'
  on-tertiary-container: '#aab2aa'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#bfedd3'
  primary-fixed-dim: '#a3d1b7'
  on-primary-fixed: '#002114'
  on-primary-fixed-variant: '#254f3b'
  secondary-fixed: '#ffdcc3'
  secondary-fixed-dim: '#e5bfa4'
  on-secondary-fixed: '#2b1706'
  on-secondary-fixed-variant: '#5b412d'
  tertiary-fixed: '#dde5dc'
  tertiary-fixed-dim: '#c1c9c0'
  on-tertiary-fixed: '#161d18'
  on-tertiary-fixed-variant: '#414942'
  background: '#effdf4'
  on-background: '#121e19'
  surface-variant: '#d8e6de'
  background-base: '#FAF9F6'
  border-subtle: '#E2E6DF'
  border-muted: '#E5E9E2'
  telemetry-active: '#1F7A4D'
  telemetry-warning: '#D97706'
  telemetry-danger: '#DC2626'
  telemetry-info: '#2563EB'
typography:
  headline-xl:
    fontFamily: Plus Jakarta Sans
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-xl-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 26px
    fontWeight: '700'
    lineHeight: 34px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  data-mono-lg:
    fontFamily: Geist
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.01em
  data-mono-md:
    fontFamily: Geist
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 18px
  data-mono-sm:
    fontFamily: Geist
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.02em
  label-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 13px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.01em
  label-sm:
    fontFamily: Plus Jakarta Sans
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 14px
    letterSpacing: 0.04em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 1.25rem
  gutter-desktop: 1.5rem
  margin: 1rem
  margin-tablet: 1.5rem
  margin-desktop: 2rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2rem
---

## Brand & Style
The design system serves an environmental acoustic monitoring platform tracking drone-assisted bioacoustic sensor nodes. It balances field-ready empirical rigor with organic calibration. The aesthetic avoids literalized ecological kitsch (no faux bamboo nodes, faux leaves, or wood grains) in favor of high-precision scientific clarity: expansive negative space, ultra-crisp tabular readouts, muted natural pigments, and understated containment.

The visual direction marries **Minimalist Data Visualization** with **Technical Precision**:
- High-legibility sans-serif typefaces paired with fixed-width telemetry readouts.
- Calm, organic earthen foundation using forest canopy greens and walnut grounds against crisp warm ivory.
- High-contrast telemetry signals (signal lock, decibel thresholds, battery, battery alerts) structured inside balanced geometric containers.
- Quiet surfaces with subtle 1px structural framing, prioritizing uninterrupted reading of real-time audio waveforms, spectrogram feeds, and device telemetry.

## Colors
The palette evokes canopy depth and scientific precision without visual noise.

- **Primary (`#214B38`)**: Canopy green anchors primary calls to action, selected navigation states, spectrogram boundary markers, and affirmative telemetry confirmations.
- **Secondary (`#795C46`)**: Walnut brown provides an organic earthy counterbalance, reserved for secondary metadata, subheadings, sensor channel categorizations, and passive timestamps.
- **Tertiary / Surface Container (`#EEF6ED`)**: Pale botanical tint used for telemetry cards, active audio frequency ranges, hover states, and container grouping.
- **Neutral (`#17231E`)**: Deep charcoal tint used for high-contrast primary typography and data values, eliminating harsh absolute blacks while maintaining WCAG AAA contrast ratios.
- **Background Base (`#FAF9F6`)**: Warm porcelain white that reduces ocular strain during extended dashboard monitoring sessions.
- **Borders (`#E2E6DF` / `#E5E9E2`)**: Crisp 1px structural rules maintaining grid alignment across tabular data and sensor modules.

## Typography
Typography is strictly sans-serif and engineered for instant empirical scanning.

- **Plus Jakarta Sans**: Applied to display titles, dashboard section headers, and prominent labels. Delivers modern structural presence without geometric sterility.
- **Inter**: Applied to analytical narratives, observational notes, and operational body copy.
- **Geist**: Deployed across telemetry stats, GPS coordinates, decibel thresholds (dBFS/SPL), flight durations, and timestamp displays.
- **Numeric Rule**: All numerical values must enable tabular figures via OpenType styling (`font-feature-settings: "tnum" 1, "cv05" 1;` or `font-variant-numeric: tabular-nums;`) to prevent layout jitter across streaming updates.

## Layout & Spacing
The layout follows a fluid-hybrid dashboard architecture. A persistent 280px left sidebar anchors spatial context across desktop screens, collapsing into an overlay navigation drawer below 1024px.

- **Desktop (≥ 1280px)**: 12-column adaptive grid layout, max-width 1680px for central telemetry panels, 24px gutters, and 32px canvas margins.
- **Tablet (768px – 1279px)**: 8-column layout with 20px gutters and 24px margins. Data grids wrap into 2x2 arrays.
- **Mobile (< 768px)**: 4-column layout with 16px gutters and 16px margins. Telemetry cards stack vertically; charts adopt horizontal scroll views for granular acoustic inspection.
- **Density**: 4px baseline rhythm. Inner card padding defaults to 20px or 24px (`space-lg`), ensuring telemetry figures remain visually separated from card boundaries.

## Elevation & Depth
Depth relies on deliberate planar stratification, subtle outlines, and low-opacity ambient light rather than aggressive drop shadows.

- **Level 0 (Canvas Base)**: `#FAF9F6`. Plain background upon which analytical surfaces rest.
- **Level 1 (Card & Modular Containers)**: Crisp solid `#FFFFFF` or pale tint `#EEF6ED` enveloped by a 1px border of `#E2E6DF`. Ambient shadow: `0 1px 3px 0 rgba(23, 35, 30, 0.04), 0 1px 2px -1px rgba(23, 35, 30, 0.02)`.
- **Level 2 (Dropdowns, Floating Controls, Telemetry Popovers)**: `#FFFFFF` encased in a 1px border of `#E2E6DF`. Ambient shadow: `0 4px 16px -2px rgba(23, 35, 30, 0.08), 0 2px 6px -1px rgba(23, 35, 30, 0.04)`.
- **Level 3 (Modal Dialogs, Spectrogram Inspect Overlays)**: `#FFFFFF` with a 1px border of `#E2E6DF` atop a translucent backdrop (`rgba(23, 35, 30, 0.4)`). Ambient shadow: `0 20px 24px -4px rgba(23, 35, 30, 0.12), 0 8px 8px -4px rgba(23, 35, 30, 0.04)`.

## Shapes
Shapes adhere to a defined curved framework (`roundedness: 2` base) scaled to component footprints:

- **Cards & Data Panels**: Rounded between `16px` and `20px` (`rounded-xl` to `rounded-2xl`). This softens dense analytical data sets and establishes clean containment.
- **Input Fields & Buttons**: Fixed at `8px` (`rounded-md` or `rounded-lg`) for clean target boundaries.
- **Badges, Telemetry Indicators & Chips**: Fully pill-shaped (`9999px`) or `6px` depending on data density. Status pills use pure rounded radii to differentiate transient metrics from interactive cards.
- **Subtle Edges**: Outer boundaries avoid stark corners while strictly refusing decorative organic curvature.

## Components

### Buttons
- **Primary**: Solid Forest green (`#214B38`) background with crisp white typography. Height 40px, padding 0 16px, 8px border-radius. Hover: `#18392A`. Active: `#122C20`.
- **Secondary**: Walnut brown (`#795C46`) outline or pale surface with walnut text. Height 40px, padding 0 16px, 8px radius. Hover: `#FAF9F6` with darkened border.
- **Tertiary / Ghost**: Transparent background with `#214B38` text. Hover: `#EEF6ED`.
- **Focus State**: 2px offset ring in `#214B38` at 30% opacity.

### Chips & Telemetry Badges
- **Status Indicators**: Pill-shaped (`rounded-full`), padding 2px 8px, font size 11px, Geist mono bold.
  - *Active Telemetry*: `#EEF6ED` surface, `#1F7A4D` text, 1px border `#C3DFBF`.
  - *Warning (e.g., High Drone Wind Noise)*: `#FEF3C7` surface, `#B45309` text, 1px border `#FDE68A`.
  - *Critical (e.g., Node Offline)*: `#FEE2E2` surface, `#B91C1C` text, 1px border `#FECACA`.
- **Filter Chips**: Height 32px, 16px border-radius. Inactive: `#FFFFFF` border `#E2E6DF` with `#795C46` text. Active: `#214B38` background with `#FAF9F6` text.

### Cards & Telemetry Containers
- **Acoustic / Data Card**: Clean white `#FFFFFF` or pale green `#EEF6ED` background, 16px–20px corner radius, 1px border in `#E2E6DF`, 20px padding.
- **Card Header**: Displays sensor node ID in Geist Mono and status badge right-aligned, separated from the card body with an optional `#E2E6DF` divider.

### Lists & Data Tables
- **Rows**: Fixed 44px (compact) or 52px (standard) heights. Border bottom 1px `#E5E9E2`.
- **Header**: Height 36px, background `#FAF9F6`, uppercase 11px label with `letter-spacing: 0.05em`, color `#795C46`.
- **Values**: Right-aligned numeric metrics rendered in Geist Mono with tabular figures enabled. Hover state on rows: `#EEF6ED` at 60% opacity.

### Form Inputs & Selectors
- **Input Fields**: Background `#FFFFFF`, border 1px `#E2E6DF`, height 40px, border-radius 8px, text `#17231E`. Placeholder in `#795C46` at 60% opacity.
- **Focus**: Border `#214B38`, subtle outer box-shadow `0 0 0 3px rgba(33, 75, 56, 0.12)`.
- **Checkboxes & Radios**: 18px size, 4px corner radius for checkboxes, full circle for radios. Active checkmark fill in `#214B38`.

### Sidebar Navigation
- **Structure**: Persistent 280px bar, background `#FAF9F6`, border-right 1px `#E2E6DF`.
- **Item**: 40px height, 8px border-radius, color `#795C46`, 14px Plus Jakarta Sans medium.
- **Active State**: Background `#EEF6ED`, text `#214B38`, font weight 600, with a 3px active indicator bar in `#214B38` on the leading edge.

### Acoustic Specialized Components
- **Audio Scrubber & Spectrogram Container**: 12px border-radius container, background `#17231E` for acoustic visualization feeds, wrapped in a 1px border `#E2E6DF`. Scrub head in `#EEF6ED` with a `#214B38` anchor point.
- **Decibel / Frequency Meter**: Horizontal or vertical bar with 4px segments. Inactive segments `#E2E6DF`; active range `#214B38` transitioning to `#D97706` past user-defined SPL thresholds.