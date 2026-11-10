# Initial data

`calibration.csv` and `test.csv` are original synthetic English examples written for this prototype, provided under CC0-1.0. They are disjoint, very small, and not representative of real social-media use. `toxic` is an approximate label for toxic language, not cyberbullying or grooming. The fixed 0.5 threshold is a research default, not calibrated against these examples. The calibration set is available for future threshold selection; the test set must not be used for tuning.

The public Dadvar YouTube archive is downloaded locally into the ignored `data/external/` directory. Its origin, terms, limitations and exploratory results are documented in `docs/research.md`. No public raw data is bundled in version control. Dataset access should not involve collection of real conversations without a separate approved research protocol.
