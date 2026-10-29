# WL57 A2 terminal measurement review

Observer formatting only; frozen decisions come from primary gate/evaluation files.

Primary source: `e809c15f60db647cf16819cd3018f52db91dad99`. Original WL57 Gate1 remains FAILED and diagnostic.

## Declared gate and outcome

```json
{
  "reference_data_readiness": "SUPPORTED",
  "direct_learned_K": "NOT SUPPORTED",
  "rule_state_learned_K": "NOT SUPPORTED",
  "rule_state_advantage": "NOT SUPPORTED",
  "absolute_fidelity": "SUPPORTED",
  "deformation_fidelity": "NOT SUPPORTED",
  "physical_contract": "SUPPORTED",
  "dynamic_cost": "INCONCLUSIVE",
  "FNTO_composition": "INCONCLUSIVE",
  "rigid": "CONDITIONALLY SUPPORTED",
  "local_deformation": "NOT SUPPORTED",
  "global_bending_twisting": "INCONCLUSIVE",
  "self_contact": "INCONCLUSIVE",
  "TransGI": "INCONCLUSIVE"
}
```

## Every fixed model/seed (worst geometry/channel)

|Model|Coarse mean relL2 max|Coarse p90 max|Fine16 mean max|Delta relL2 max|Delta correlation min|Useful+physical|
|---|---:|---:|---:|---:|---:|---|
|D_5701|0.0704053|0.118366|0.114752|0.663437|0.765152|False|
|D_5702|0.0705494|0.117024|0.115766|0.65793|0.753476|False|
|R_5701|0.0676195|0.115355|0.112491|0.624732|0.821575|False|
|R_5702|0.0678361|0.114674|0.112619|0.599247|0.8335|False|

All geometry/channel/field and angular/noise rows remain in heldout.json and operator_metrics.csv; the worst-case view above does not replace them.

## Fair best-cache lifecycle, including common geometry (seconds)

|Pose|Seed|Queries|D|R|R/D|D policy|R policy|
|---|---:|---:|---:|---:|---:|---|---|
|cf0|5701|1|0.340036|0.343196|1.00929|full|compact|
|cf0|5701|4|0.344014|0.347899|1.01129|full|full|
|cf0|5701|16|0.359934|0.363814|1.01078|full|full|
|cf0|5701|64|0.432615|0.433288|1.00156|full|full|
|cf0|5702|1|0.340037|0.343916|1.01141|full|compact|
|cf0|5702|4|0.344005|0.347902|1.01133|full|full|
|cf0|5702|16|0.359946|0.363793|1.01069|full|full|
|cf0|5702|64|0.43154|0.436173|1.01074|full|full|
|cf1|5701|1|0.340722|0.343807|1.00906|full|compact|
|cf1|5701|4|0.344703|0.348576|1.01124|full|full|
|cf1|5701|16|0.360627|0.364459|1.01063|full|full|
|cf1|5701|64|0.434163|0.435228|1.00245|full|full|
|cf1|5702|1|0.340708|0.344611|1.01146|full|compact|
|cf1|5702|4|0.344696|0.34857|1.01124|full|full|
|cf1|5702|16|0.360593|0.364479|1.01078|full|full|
|cf1|5702|64|0.432936|0.435038|1.00486|full|full|
|cf2|5701|1|0.340261|0.343953|1.01085|full|compact|
|cf2|5701|4|0.344234|0.348126|1.01131|full|full|
|cf2|5701|16|0.360163|0.364028|1.01073|full|full|
|cf2|5701|64|0.435055|0.438212|1.00726|full|full|
|cf2|5702|1|0.340255|0.34415|1.01145|compact|full|
|cf2|5702|4|0.344228|0.348117|1.0113|full|full|
|cf2|5702|16|0.360131|0.364053|1.01089|full|full|
|cf2|5702|64|0.435205|0.435731|1.00121|full|full|
|cf3|5701|1|0.341009|0.3449|1.01141|full|compact|
|cf3|5701|4|0.344997|0.34889|1.01129|full|full|
|cf3|5701|16|0.360934|0.364841|1.01083|full|full|
|cf3|5701|64|0.434214|0.433745|0.998919|full|full|
|cf3|5702|1|0.341019|0.34484|1.0112|compact|compact|
|cf3|5702|4|0.344999|0.348872|1.01123|full|full|
|cf3|5702|16|0.360915|0.364791|1.01074|full|full|
|cf3|5702|64|0.43616|0.432894|0.992514|full|full|

A lower ratio is not an architecture advantage unless both arms pass all frozen quality/physical conditions and the>=25% noninferiority/memory/timing rule. Oracle data generation and training are offline costs, separately retained in stage attempt logs.

## Conditional composition

Not executed unless the fixed learned operator gates permit it. No coupled physical-GI success is inferred.


## Interpretation limits and four geometry cases

Rigid material-frame reuse was tested in A1. Local dents are the sole primary deformation family. Global bending/twisting and self-contact are untested; no sparse/local/low-rank change assumption is validated. A benefit, if present, concerns this concrete learned nonlinear graph-state mechanism against matched-context direct conditioning, not abstract Rule-State labels. Old training targets retain their original Monte Carlo noise; new independent confirmation uses unbiased illumination importance sampling. Distinguish data/reference noise, training/optimization, representation bias, geometry generalization and real lifecycle cost. No TransGI superiority, production adoption or WL58 follows.

