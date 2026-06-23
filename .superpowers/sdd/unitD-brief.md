### Task 9: Traditional-Chinese plans + relocate the agentbot plan (user Task 4)

**Files:**
- Create: `docs/plans/2026-06-21-vlm-brain-deployment.zh-Hant.md`
- Move + translate: `docs/plans/2026-06-21-vlm-brain-agentbot-integration.md` → `../../agentbot/docs/plans/2026-06-21-vlm-brain-agentbot-integration.zh-Hant.md` (it is agentbot-scoped — its target file structure is under `agentbot/`; see [[agentbot-scaffold]]).

- [ ] **Step 1: Translate the deployment plan** (superseded draft) to Traditional Chinese, preserving headings, code blocks, file paths, and the SUPERSEDED banner verbatim (do not translate code/paths/identifiers). Keep the English `# …Implementation Plan` H1 with a `（繁體中文）` suffix.

- [ ] **Step 2: Translate + move the integration plan** into `agentbot/docs/plans/`. Leave a one-line pointer stub at the old path: `> 已移至 agentbot/docs/plans/…（繁體中文）`.

- [ ] **Step 3: Verify** both files exist and the code fences/line counts match the originals (translation must not drop fenced blocks):

Run: `grep -c '```' docs/plans/2026-06-21-vlm-brain-deployment.zh-Hant.md` (compare to original's fence count).

- [ ] **Step 4: Stage** — `git add docs/plans/ ../../agentbot/docs/plans/` (suggested: `docs(plans): zh-Hant translations; move integration plan to agentbot`)

---

## Self-review

- **Spec coverage:** Task 1 VQA replan → Tasks 1-7 (6 phases ✓, 8B teacher ✓, 7 types ✓, 4090 ✓, feasibility ✓). User Task 2 (architecture) → answered in chat + diagram + Background section. User Task 3 (script HTML) → Task 8. User Task 4 (zh-Hant plans + relocation) → Task 9.
- **Placeholders:** none — all code/commands are concrete; doc tasks specify exact sections + verification greps.
- **Type consistency:** `_PHASES`, `_PHASE_ORDER`, `_sample_phase_frames`, `template_qa_pairs`, `generate(..., phase_mode=)`, `generate_qa(image, task, phase, …)` are used consistently across Tasks 1-3 and 5.
- **Risk:** select_layer 16 vs 28-layer LoRA (documented in Task 7 note); teacher VRAM (8B fits, 4B fallback documented in Task 4).
