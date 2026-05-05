# COLA folder organization note (2026-04-01)

Location: /home/qiankun/CommRL

Current COLA-related folders:
- COLA: active working repository, larger and contains research/code/plots/logs plus local experiment outputs.
- COLA_SemCom_RA_export: cleaner export package for SemCom RA, already organized for sharing/export.

What was organized today:
- Moved root-level result text files from COLA/ into COLA/results/.
- Kept source code and existing git history untouched.
- Did not rewrite training scripts or runtime paths.

Important note:
- Future runs of COLA may still recreate *_obj_list.txt and *_ep_obj_list.txt in the COLA root, because agent.py currently writes those files to the project root.
- If needed, the next cleanup step should be updating agent.py so new outputs are written directly into COLA/results/.
