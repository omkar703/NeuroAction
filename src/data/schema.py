from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class CanonicalTrial:
    """Canonical multimodal trial representation for NeuroAction.
    
    Harmonizes pre-movement EEG (200 Hz, 1-second intention window),
    movement-execution EEG (100 Hz), 3D hand kinematics (Leap Motion, 100 Hz),
    and external visual ball state (100 Hz).
    
    Standardized Trial Indexing:
    - trial_index: 0-based Python array index (0, 1, 2, ...)
    - trial_number: 1-based experimental trial number (trial_index + 1)
    - eeg_seg_idx: Original MATLAB 1-based index from derivative files
    """
    # Identifiers & Standardized Indexing
    participant_id: str
    run_id: str
    trial_id: str
    trial_index: int                       # 0-based Python array index
    trial_number: int                      # 1-based experimental trial number (trial_index + 1)
    session_id: Optional[str] = None
    eeg_seg_idx: Optional[int] = None      # Original MATLAB 1-based reference from derivative
    
    # Experimental State & Semantics (Strictly Separated Fields)
    label_code: int = -1
    target_goal: str = ""                  # "left" or "right" (intended abstract reaching target)
    physical_hand_direction: str = ""      # "left" or "right" (physical motor execution)
    ball_direction: str = ""               # "left" or "right" (external machine / screen movement)
    ball_color: str = ""                   # "red" (congruent) or "yellow" (incongruent)
    congruency: str = ""                   # "congruent" or "incongruent"
    
    # Pre-Movement EEG
    eeg_pre: Optional[np.ndarray] = None    # shape: (n_samples_pre, n_channels)
    times_pre: Optional[np.ndarray] = None  # seconds: -1.0 to 0.0 relative to onset
    srate_pre: float = 200.0
    channels: Optional[List[str]] = None    # Canonical 26 scalp EEG channels
    
    # Movement-Execution EEG
    eeg_me: Optional[np.ndarray] = None     # shape: (n_samples_me, n_channels_raw), e.g. 26 or 28
    times_me: Optional[np.ndarray] = None   # ms: -200 ms to target hit
    srate_me: float = 100.0
    channels_me: Optional[List[str]] = None # Original channels in execution EEG (26 or 28 with M1/M2)

    @property
    def target_direction(self) -> str:
        """Alias for target_goal."""
        return self.target_goal

    @property
    def hand_direction(self) -> str:
        """Alias for physical_hand_direction."""
        return self.physical_hand_direction

    @property
    def eeg_me_scalp(self) -> Optional[np.ndarray]:
        """Return execution EEG projected onto canonical 26 scalp channels (excluding M1, M2 if present)."""
        if self.eeg_me is None:
            return None
        if self.eeg_me.shape[1] == 26:
            return self.eeg_me
        elif self.eeg_me.shape[1] == 28:
            # Scalp indices: exclude indices 18, 19 (M1, M2)
            scalp_idx = [i for i in range(28) if i not in (18, 19)]
            return self.eeg_me[:, scalp_idx]
        return self.eeg_me
    
    # Physical Motor Kinematics (Leap Motion)
    hand_x: Optional[np.ndarray] = None     # mm (lateral axis: left < 0, right > 0)
    hand_y: Optional[np.ndarray] = None     # mm (vertical height)
    hand_z: Optional[np.ndarray] = None     # mm (depth)
    
    # External Machine / Visual Ball State
    ball_x_pix: Optional[np.ndarray] = None # pixels (screen space)
    
    # Quality & Alignment Metadata
    valid: bool = True
    coverage: float = 0.0                   # coverage ratio 0.0 - 1.0
    start_res_ms: float = 0.0               # temporal alignment residual (ms)
    is_multimodal: bool = True              # True if both pre-eeg and execution present
    qc_notes: str = ""

    def __repr__(self) -> str:
        pre_shape = self.eeg_pre.shape if self.eeg_pre is not None else None
        me_shape = self.eeg_me.shape if self.eeg_me is not None else None
        return (
            f"CanonicalTrial({self.trial_id}, number={self.trial_number}, cond={self.congruency}, "
            f"goal={self.target_goal}, hand={self.physical_hand_direction}, ball={self.ball_direction}, "
            f"cov={self.coverage:.2f}, pre_shape={pre_shape}, me_shape={me_shape})"
        )
