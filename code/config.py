"""Settings shared by every script: paths, signal format, training budget, target chip and the AHA targets."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "physionet"
CLIPS = DATA / "clips.npz"
RESULTS = ROOT / "results"

# Signal: one ECG lead, 5 s clips at 125 Hz = 625 samples per clip
FS, WIN_S = 125, 5
WIN = FS * WIN_S
DATABASES = ("cudb", "vfdb", "mitdb")          # v1 sources: rhythm-annotated recordings
EXTRA = ("challenge-2015_training", "sddb")      # v2 sources: ICU arrhythmia alarms, sudden-death Holter
CAP = 60  # at most this many clips (5 min) per person per rhythm category, spread evenly over the recording

# Training
SEED, FOLDS = 0, 5
EPOCHS, FT_EPOCHS, BATCH = 40, 20, 128
N_CALIB = 300  # clips used to calibrate full-integer int8 (from training patients only)

# Compression settings
KEEPS = (0.75, 0.50, 0.35)    # structured pruning: fraction of filters kept in every layer
SCRATCH_KEEPS = (0.50, 0.35)  # same narrow shapes, trained from scratch (the "train small" control)
SPARSITY = 0.75               # unstructured pruning: fraction of weights set to zero in every layer

# Target chip: STM32 NUCLEO-L432KC (Arm Cortex-M4, 80 MHz)
FLASH_BYTES, RAM_BYTES, MCU_HZ = 256 * 1024, 64 * 1024, 80e6
MACS_PER_CYCLE = (0.5, 1.0)   # assumed int8 throughput range with CMSIS-NN kernels (estimate, not measured)

# AHA performance goals for AED rhythm analysis (Kerber et al., Circulation 1997;95:1677).
# (rhythm category, metric, goal, lowest acceptable one-sided 90% lower confidence limit)
AHA = [
    ("coarse VF", "sensitivity", 0.90, 0.87),
    ("rapid VT", "sensitivity", 0.75, 0.67),
    ("NSR", "specificity", 0.99, 0.97),
    ("other non-shockable", "specificity", 0.95, 0.88),
    ("asystole", "specificity", 0.95, 0.92),
]
REPORT_ONLY = ["fine VF", "other VT"]  # "intermediate" rhythms: the AHA asks for the result but sets no goal

# Rhythm labelling thresholds. VF amplitudes follow Kerber 1997. The AHA leaves the rapid-VT rate cut-off
# to the manufacturer; 180 bpm is our choice.
COARSE_VF_UV, ASYSTOLE_UV, RAPID_VT_BPM = 200, 100, 180
