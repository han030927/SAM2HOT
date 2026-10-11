# Third-party code

The root MIT license does not replace the original terms of third-party code.

- `dam4sam/sam2/`: DAM4SAM's SAM2.1 implementation, obtained through SENTRY.
  Original copyright headers are retained. DAM4SAM currently provides the
  Apache-2.0 license at
  https://github.com/jovanavidenovic/DAM4SAM/blob/master/LICENSE;
  a copy is included at `dam4sam/LICENSE`.
- SENTRY attribution and license text are retained in `licenses/`.
  Its notice dated 2026-07-15 predates DAM4SAM's currently published license.
  The historical notice is kept unmodified.
- SAM2-derived code retains Meta's copyright and Apache-2.0 terms.

The SAM2HOT wrappers adapt tracking, band regrouping, motion recovery and data
loading to the selected HOTC2020 setting. The bundled native DAM4SAM/SAM2 files
are unchanged from the source copy used for validation.
