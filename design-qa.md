# Sona design QA

## Visual truth and evidence

- Source visual truth: `/Users/karangathani/Documents/GitHub/local-voice-cloning/design-reference.png` (1,487 × 1,058 px).
- Desktop implementation, completed state: `/Users/karangathani/Documents/GitHub/local-voice-cloning/implementation-desktop-final-normalized.png` (1,440 × 1,024 px).
- Desktop implementation, live generation state: `/Users/karangathani/Documents/GitHub/local-voice-cloning/implementation-desktop-active-normalized.png` (1,440 × 1,024 px).
- Side-by-side comparison: `/Users/karangathani/Documents/GitHub/local-voice-cloning/design-comparison-final.png` (source, active implementation, completed implementation).
- Mobile implementation: `/Users/karangathani/Documents/GitHub/local-voice-cloning/implementation-mobile-final.png` at a 390 × 844 CSS viewport.
- Browser screenshots were captured at device pixel ratio 2. The desktop captures were cropped to the browser's CSS frame and normalized to 1,440 × 1,024 before comparison.

## Full-view comparison

The implementation preserves the selected concept's dark local-studio palette, three-band hierarchy, split script/reference workspace, amber primary action, mint progress language, compact technical metadata, and persistent generated-output area. The source combines an in-progress Voice stage with a ready output, so the implementation was compared in two truthful states: live generation for transport/progress fidelity and completed generation for playback/download fidelity.

The Qwen build intentionally replaces the source's legacy speed/step controls with model quality and output-language controls. It also uses native accessible audio players instead of drawing a decorative waveform. Those differences follow the implemented engine and keep every visible control functional.

## Focused regions

- Header: brand, local-privacy message, and MLX engine badge retain the source hierarchy and alignment.
- Script/reference: loaded-reference metadata, 12-second profile-window explanation, exact transcript override, and quality result are visible and readable.
- Progress: Prep, Load, Voice, and Finish are driven by backend events; active, complete, error, and idle states were exercised.
- Output: a real generated result exposes playback plus WAV and MP3 downloads.

## Comparison history

1. Initial desktop comparison: core layout and visual system matched. The first capture predated the final visible quality/language controls, so it was not accepted as final evidence (P2).
2. Current desktop active/success comparison: no P0, P1, or P2 visual defects remained at 1,440 × 1,024.
3. Responsive pass: a 21 px horizontal overflow was found at 390 px because the delivery-control grid retained the select input's 300 px intrinsic width (P1).
4. Responsive fix: the controls now collapse to a `minmax(0, 1fr)` track and force their Shiny input container to 100% width. Recheck measured `scrollWidth = clientWidth = 390`; tablet and desktop also measured zero horizontal overflow.

## Interaction and runtime verification

- Uploaded a real 7.7-second, 24 kHz mono WAV reference.
- Confirmed the reference quality checks and metadata.
- Entered the exact reference transcript and ran the BF16 Qwen3-TTS model end to end three times after caching.
- Observed backend-driven Prep → Load → Voice → Finish progress and elapsed time.
- Confirmed the final player and both download actions.
- Confirmed the normal browser preview remains connected with a real playable result open.
- Browser inspection found no application console errors; Shiny/Bootstrap emitted only their known datepicker deprecation warnings during earlier inspection.
- Automated verification: 45 tests passed (5 integration tests deselected), Ruff passed, dependency dry-run required no changes, and `git diff --check` passed.

final result: passed
