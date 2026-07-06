"""BCI (Brain-Computer Interface) module.

Cross-subject prototypes for the "Hugging Face of BCI" vision:
- datasets: multi-subject EEG/MEG loading
- preprocessing: common filter + epoch pipeline
- models: small cross-subject transformer
- training: leave-one-subject-out evaluation
- neuromodulation: ACh/NE/DA-conditioned decoder wrapper (Dayan-correction extension)
"""
