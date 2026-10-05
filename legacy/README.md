# Historical v1 executables

Do not run; reads test and combined data. These files are retained for historical audit only. The current workflow is `python -m models.run_all` and uses guarded train/validation readers. No guarantees of portability, privacy suppression, or test isolation apply to legacy scripts.

`p3_parser_original.py` preserves P3’s untouched proposal for audit only. Never import or execute it: its outcome/unsupported-condition mapping and online model loading are superseded by backend/parser.py. Historical temporal/clinical wording in this archived source must not be used in presentation claims.
