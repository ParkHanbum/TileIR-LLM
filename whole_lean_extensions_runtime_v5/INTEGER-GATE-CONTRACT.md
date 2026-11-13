# Integer72 integration contract (source implemented, global execution blocked)

Actual typed integer72 model proof and frozen lean binding are available; the new global endpoint remains pending. Existing v4 source remains unchanged.

The inherited lean26_gdn_finite_runtime_v1/gates.py dispatch lines64–65 require retained_integer_site_counts/retained_integer_checks=72. The typed core correctly replaces these with typed_integer_* and reports native integer predicates0 / stream synchronizations72. A new global gate cannot simply pass this candidate receipt to the old native-integer assertion, nor fabricate retained native counts.

The new source-bound gate explicitly validates typed per-site24, total72, native predicates0 and stream synchronizations72, while preserving original float360/site counts/descriptors/other component assertions. integer_gate.py constructs a separate source-owned dispatch variant and propagates it through owned audit globals. It does not mutate predecessor modules or translate typed receipts into native evidence. Negative regression rejects native72 labels, missing synchronization count, per-site mismatch and changed float checks.

Installation must preserve existing finite+EQ GDN body identity and callback globals. No override of the fresh owned body by a second independent copy. Actual model/combined endpoint proof needed after all sampling144/temp16 prerequisites pass. Profile remains blocked until global endpoint/exit0/recovery.

Final update: actual individual model proofs and final lean sources are now present. Corresponding v5 glue is implemented; only the new global endpoint/profile remain pending. Earlier planning text above is historical rationale, not current proof status.
