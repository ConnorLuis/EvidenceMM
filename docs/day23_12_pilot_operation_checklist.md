\
# Day23 12-Pilot Operation Checklist

Use the same recorder command and the dedicated output directory
`outputs\episodes_root_cause_v2_pilot_day23`.

Before each row, reset the scene according to the matched group and predeclare
the single primary intervention.

| Order | Row | Level | Operation |
|---:|---|---|---|
| 1 | p23_g01_clean | control | Nominal pick-place; no intervention. |
| 2 | p23_g01_target | mild | Move red cube from nominal marker in the chosen safe direction by measured mild offset; follow original nominal grasp path and do not compensate. |
| 3 | p23_g01_gripper | mild | Nominal object/path; intentionally shift grasp-close in the fixed early/late direction at mild level. |
| 4 | p23_g01_trajectory | mild | Nominal object/gripper timing; introduce one mild measured lateral approach deviation toward free space. |
| 5 | p23_g02_clean | control | Reset scene; nominal pick-place; no intervention. |
| 6 | p23_g02_target | medium | Same target-offset direction; larger measured offset than G01; do not compensate. |
| 7 | p23_g02_gripper | medium | Same early/late direction; larger timing shift than G01; path otherwise nominal. |
| 8 | p23_g02_trajectory | medium | Same free-space deviation direction; larger measured waypoint offset than G01. |
| 9 | p23_g03_clean | control | Reset scene; nominal pick-place; no intervention. |
| 10 | p23_g03_target | strongest-safe | Same target-offset direction; largest still-safe measured pilot offset; do not compensate. |
| 11 | p23_g03_gripper | strongest-safe | Same early/late direction; largest still-safe pilot timing shift; path otherwise nominal. |
| 12 | p23_g03_trajectory | strongest-safe | Same free-space direction; largest still-safe measured pilot path deviation. |

After every run:

1. copy the timestamp episode folder name into `episode_id`;
2. record recorder `OVERALL EPISODE` result;
3. if FAIL, copy failed check names;
4. record actual task success/failure;
5. record whether the declared intervention was applied;
6. record the numeric parameter where manually measurable;
7. review front/wrist/state-action observability;
8. record any unexpected contact, safety abort, or hardware fault;
9. retain every raw attempt, including failed interventions.

Do not silently replace or delete a pilot row.
