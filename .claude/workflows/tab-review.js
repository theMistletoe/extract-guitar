export const meta = {
  name: 'tab-review',
  description: 'Check the tab against the guitar stem (errors, missing notes, timing, reproduction), have flagged spots judged from two perspectives, apply the calibrated rule and regenerate',
  whenToUse: 'After scripts/transcribe_tab.py: to find and fix wrong or missing notes in outputs/target/tab by comparing the rendered tab with the extracted guitar. Costs about two agents per 15 flagged spots.',
  phases: [
    { title: 'Measure', detail: 'render the tab, compare it with the stem (compare_tab_audio.py), pack the flagged spots (review_tab.py pack)' },
    { title: 'Judge', detail: 'each batch judged independently by an acoustic analyst and an adversarial skeptic' },
    { title: 'Apply', detail: 'review_tab.py collect (calibrated rule) and regenerate tab, checks, checker, audio, comparison' },
  ],
}

// args (all optional): { python, python_bp, thrNote, thrMissing, apply }
//   python     command for the project environment (default: uv run --no-sync python)
//   python_bp  command for an environment that also has Basic Pitch (default: python); the calibration
//              in reports/tab_compare_calibration.json was fitted with Basic Pitch features
//   thrNote / thrMissing  review tab notes with P(wrong) / candidates with P(real) at or above (0.3)
//   apply      false: only judge, do not edit the tab
const A = args || {}
const PY = A.python || 'uv run --no-sync python'
const PYBP = A.python_bp || PY
const thrNote = A.thrNote ?? 0.3
const thrMissing = A.thrMissing ?? 0.3

const PACK = {
  type: 'object',
  properties: {
    folder: { type: 'string' }, work: { type: 'string' }, ids: { type: 'array', items: { type: 'string' } },
    items: { type: 'number' }, tab_notes: { type: 'number' }, missing: { type: 'number' },
    summary: { type: 'object', description: 'reproduction and timing blocks of outputs/target/tab/compare/summary.json' },
  },
  required: ['folder', 'work', 'ids', 'items'],
}
const VERDICTS = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          verdict: { type: 'string', enum: ['correct', 'wrong', 'real', 'not_real'] },
          p: { type: 'number', description: 'tab_note: probability the tab note is wrong; missing: probability the note is really played' },
          fix: { type: 'string', description: 'wrong tab_note: remove | octave_up | octave_down | pitch:<midi> | timing; real missing: add; otherwise none' },
          reason: { type: 'string', description: 'one or two sentences citing the evidence' },
        },
        required: ['id', 'verdict', 'p', 'fix', 'reason'],
      },
    },
  },
  required: ['verdicts'],
}

phase('Measure')
const packed = await agent(`Work in the repository root (your working directory). Run these commands in order and stop at the first failure (report its error output instead):
1. Only if reports/tab_compare_calibration.json does not exist: ${PYBP} scripts/compare_tab_audio.py --bench   (needs data/bench_tab from scripts/bench_tab.py)
2. ${PYBP} scripts/compare_tab_audio.py
3. ${PY} scripts/review_tab.py pack --thr-note ${thrNote} --thr-missing ${thrMissing}
Return the JSON object printed by step 3, and as "summary" the "reproduction" and "timing" blocks of outputs/target/tab/compare/summary.json.`,
  { label: 'measure', phase: 'Measure', schema: PACK })
if (!packed || !packed.ids.length) {
  log('nothing to review')
  return { packed }
}
log(`${packed.items} spots to review (${packed.tab_notes} tab notes, ${packed.missing} possibly missing) in ${packed.ids.length} batches`)

const PROBE = `${PYBP} scripts/compare_tab_audio.py probe ${packed.folder}`
const COMMON = (batch, label, lens) => `You are checking an automatic guitar transcription (a tab) against the recording without being able to listen. You judge a batch of items using only cached analysis in one folder.

Folder: ${packed.folder}
- "stem": the guitar separated from a band recording (the evidence of what was played)
- "rendering": the tab played back by a sampled guitar at the recorded times (what the tab says)
- "residual": the other instruments that were separated away (a source of leakage into the stem)
- notes.csv: every tab note (onset_s, offset_s, pitch, string 1 = high e .. 6 = low E, fret, bar, beat, sixteenth)

Tool (run with Bash from the repository root; it is fast, loop over items freely):
  ${PROBE} --t <seconds> --pitch <midi> [--image ${packed.work}/${label}/<name>.png]
It prints, for the queried pitch and its harmonics 1-6 (+0, +12, +19, +24, +28, +31 semitones): the level in dB at the onset and, in brackets, its rise over the preceding 150 ms, for stem / rendering / residual. Then the onset posteriors (0-1, max within 40 ms) of three transcription models on the stem at pitch-12, pitch, pitch+12, their two-model average, and the same average on the rendering. Then tab notes and Basic Pitch notes within 250 ms (time offsets relative to the query). With --image it writes a zoomed constant-Q picture (stem | rendering | residual; white boxes = tab notes; cyan = the queried pitch and its harmonics) that you can look at with the Read tool. Create ${packed.work}/${label}/ first if you save images.

Rules: use only the probe command, the folder above, your items file and your own image folder.

Items: read ${packed.work}/items_${batch}.json (a JSON list). Two types:
- tab_note: a note written in the tab (pitch, string, fret, time). They were flagged by a statistical check as possibly wrong, and many flags are false alarms. Decide 'correct' (the guitar in the stem really plays this pitch at about this time) or 'wrong'. If wrong, give the fix: remove | octave_up | octave_down | pitch:<midi> (a neighbouring pitch is played instead) | timing (the same pitch is played but more than 50 ms away).
- missing: a pitch/time where some detector heard a note that the tab lacks. Decide 'real' (the guitar plays a new note of this pitch here, so the tab should add it; fix 'add') or 'not_real' (an overtone of another sounding note, leakage from the other instruments, a note that is still ringing from earlier, or noise).
Feature fields (strings): post_ens = two-model average onset posterior at the pitch; votes = how many of three models reach 0.3; bp = Basic Pitch detects it; alt_octave = posterior at the octave minus at the pitch (positive: the octave looks likelier); overtone = the pitch is a harmonic of a tab note already sounding; reattack = the same pitch is still sounding from an earlier tab note; flux_s / flux_r = onset rise (dB) at the fundamental in stem / rendering; resid = residual minus stem level at the pitch (dB; positive means the other instruments are louder there); for missing items also sources, n_models, post_max, harmonic, octave (a tab note an octave away at the same time), same_pc, same_pitch_tab_ms (nearest tab note of the same pitch, ms).
Musical context fields: bar, beat, sixteenth, chord (the chord symbol estimated there from the tab; it may itself be affected by errors) and repeats_of_this_bar (bars whose notes largely repeat this bar). The same player usually repeats a passage the same way, so you may probe the same beat/sixteenth in a repeat bar (find its time in notes.csv) as supporting evidence; it is not proof.

Give p as a calibrated probability that the suspicion holds (tab_note: the note is wrong; missing: the note is really played), consistent with your verdict (p >= 0.5 exactly when the verdict is 'wrong' or 'real'). Return one verdict per item, every item, using the ids given. Before returning, also write the same {"verdicts": [...]} object as JSON to ${packed.work}/verdicts_${lens}_${batch}.json.`

const LENSES = {
  acoustic: `Your perspective: acoustic analyst. Work from the signal. A played guitar note shows a rise at its fundamental and at several of its harmonics at the onset in the stem, much as the rendering shows for the tab's own notes (compare rises rather than absolute levels: the rendering's timbre differs). A pitch that only shows energy at frequencies that are harmonics of another sounding note is an overtone, not a note; check the octave below and the fifth below. Energy that is as strong or stronger in the residual at the same time and pitch is leakage from the other instruments. Weigh the models' posteriors but do not simply copy them; look at the picture when the numbers are ambiguous.`,
  skeptic: `Your perspective: adversarial skeptic. For every item first argue against the suspicion: for a tab_note, look for evidence that it IS played (rises at the fundamental and harmonics, a chord shape that needs it, a model that hears it, the same note in a repeat bar); for a missing item, look for evidence that it is NOT a new guitar note (overtone, leakage, ringing, noise, a tab note of the same pitch nearby that already covers it). Only give 'wrong' or 'real' when the evidence clearly overrides your counter-argument. Also use the musical context: consistent chord shapes, doubled or implausible voicings on a guitar, the chord symbol, repeats.`,
}

phase('Judge')
const judged = (await parallel(packed.ids.flatMap(b => Object.entries(LENSES).map(([lens, text]) => () =>
  agent(`${COMMON(b, `${lens}_${b}`, lens)}\n\n${text}`, { label: `${lens}_${b}`, phase: 'Judge', schema: VERDICTS })
    .then(r => (r ? { batch: b, lens, verdicts: r.verdicts } : null)))))).filter(Boolean)
log(`${judged.length}/${2 * packed.ids.length} judgements returned`)
if (judged.length < 2 * packed.ids.length) log('some batches lack a verdict: collect will need --partial or a re-run')
if (A.apply === false) return { packed, judged }

phase('Apply')
const applied = await agent(`Work in the repository root. Run these commands in order and stop at the first failure (report its error output instead):
1. ${PY} scripts/review_tab.py collect${judged.length < 2 * packed.ids.length ? ' --partial' : ''}
2. ${PY} scripts/transcribe_tab.py
3. ${PYBP} scripts/verify_tab.py
4. ${PY} scripts/make_tab_check.py
5. ${PY} scripts/render_tab_audio.py
6. ${PYBP} scripts/compare_tab_audio.py
Return a short JSON-like summary: what step 1 printed, the "edits" and "notes" fields of outputs/target/tab/frevo_guitar_tab.json, and the "reproduction", "timing" and "expected_wrong_notes" fields of outputs/target/tab/compare/summary.json.`,
  { label: 'apply', phase: 'Apply' })
return { packed, judged: judged.length, applied }
