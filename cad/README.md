# WillSpeak case: CAD models

The printed parts of the case, version 12. `stl/` is for a slicer, `step/` for CAD.

| Part | File | Copies | How it prints |
|---|---|---|---|
| Case body | `pi_case_body` | 1 | as exported, open top up |
| Lid | `pi_case_lid` | 1 | face down, so the name forms on the bed |
| Lid key | `lid_key` | 1 | lying flat |
| Spacer | `hat_spacer_print_4` | 4 | standing up, with a brim |
| Rail plate | `rail_plate` | 1 | as exported |
| Battery tray | `battery_tray` | 1 | as exported |

One colour per part, 0.2 mm layers, no supports. `step/assembly.step` holds every part in place.

The tray is built for a 65 W, 20,000 mAh power bank of 109.6 x 71.0 x 29.0 mm.
Measure your own power bank before printing the tray and the rail plate.
