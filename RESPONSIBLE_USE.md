# Responsible use

voicesmith makes convincing copies of real people's voices. Used well, that gives someone their voice back after an illness, lets a creator voice their own videos without recording every line, or lets a team prototype with a colleague's blessing. Used badly, it is fraud, harassment, or putting words in someone's mouth.

## The rules the software enforces

- **No consent, no audio.** A voice cannot render until it has a consent record: a spoken statement with a one-off code, read by the speaker and checked by speech recognition and voice match, or a written attestation by the operator naming who authorised it and how.
- **Spoken consent must match the voice.** If the person who read the consent statement does not sound like the voice's references, rendering is blocked.
- **Agents cannot grant consent.** The MCP server can read consent status and tell a human what to do. It has no tool that creates or edits consent.
- **Every output is marked.** An AudioSeal watermark is embedded in every file, Chatterbox engines add their own PerTh mark, and every file carries disclosure tags saying it is AI-generated and was not recorded live. Delivered files are decoded and re-checked; if the watermark or tags are missing, the file is deleted rather than delivered.
- **Revocation works.** `voicesmith consent revoke <voice>` stops the voice rendering immediately.

## What the software cannot enforce

This is MIT-licensed code. Anyone can fork it and delete every one of these checks, and a watermark can be weakened by heavy editing. The safeguards are there to make the honest path the easy one and to leave evidence, not to stop a determined bad actor. That part is on you.

## Do not

- Clone anyone without their clear, informed permission.
- Present generated audio as a real recording, or remove its watermark or disclosure.
- Use a cloned voice to deceive, defraud, harass, or impersonate, including in phone calls, voice notes, or "it's me, I've lost my phone" messages.
- Clone children, or anyone who cannot meaningfully consent.

## The law

Several jurisdictions now regulate synthetic voices: the EU AI Act's transparency duties (Article 50, applying from 2 August 2026, with no open-source exemption), the Tennessee ELVIS Act, California AB 2602 and AB 1836, China's AI labelling measures, and South Korea's AI Basic Act among them. The US NO FAKES Act was still a bill when this was written. Check what applies to you. This is not legal advice.

## Reporting misuse

If you find voicesmith being used to harm someone, open an issue at https://github.com/fergo5002/voicesmith/issues (or, for anything sensitive, use GitHub's private vulnerability reporting on the repository).
