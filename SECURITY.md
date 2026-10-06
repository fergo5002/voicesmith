# Security and misuse

## Reporting a vulnerability

Use GitHub's private vulnerability reporting on this repository (Security tab, "Report a vulnerability"). Please include what you found, how to reproduce it, and what it affects.

These count as security issues here, not just ordinary bugs:

- any way to render audio for a voice without a usable consent record,
- any way for an agent using the MCP server to create, change, or bypass consent,
- any way a delivered file can be missing its watermark or disclosure tags,
- revoked consent that does not stop rendering,
- a path traversal or code execution route through voice names, media files, URLs or model downloads.

## Reporting misuse

If you believe voicesmith has been used to imitate you or someone else without consent, the watermark can help show where audio came from: run `voicesmith verify <file>`. Report misuse to the platform where the audio appeared and, where it is a crime, to the police. You can also open an issue here; we cannot trace individual users, because voicesmith runs on their own machines and sends us nothing.
