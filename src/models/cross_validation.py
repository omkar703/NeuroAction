"""Cross-validation and experiment orchestration engine for Phase 2.

Implements:
1. Strict Leave-One-Subject-Out (LOSO) cross-validation with zero data leakage.
2. Goal Decoder (Target A) vs Motor-Command Decoder (Target B).
3. The Critical Cross-Condition Transfer experiment matrix.
4. Temporal sliding-window decoding across [-1000, 0] ms.
5. Statistical Neural Commitment Point (NCP) detection.
6. Motor-focused channel ablation analysis.
7. Out-of-fold trial-level prediction export.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.data.schema import CanonicalTrial
from src.features.pre_eeg import (
    get_sliding_windows,
    extract_trial_features,
    MOTOR_FOCUSED_CHANNELS,
)
from src.models.decoders import (
    ChanceBaseline,
    LogisticRegressionDecoder,
    LDADecoder,
    CSPLDADecoder,
    CSPLogRegDecoder,
)
from src.models.eegnet import EEGNetClassifier


class ExperimentRunner:
    """Orchestrates Phase 2 experiments across subjects, windows, and conditions."""

    def __init__(self, trials: List[CanonicalTrial]):
        self.trials = [t for t in trials if t.valid]
        if not self.trials:
            raise ValueError("No valid trials provided to ExperimentRunner.")

        self.subjects = sorted(list(set(t.participant_id for t in self.trials)))
        self.sliding_windows = get_sliding_windows()

    def _get_model(self, model_name: str, random_state: int = 42):
        if model_name == "chance":
            return ChanceBaseline()
        elif model_name == "logistic_regression":
            return LogisticRegressionDecoder(C=1.0, random_state=random_state)
        elif model_name == "lda":
            return LDADecoder(shrinkage="auto")
        elif model_name == "csp_lda":
            return CSPLDADecoder(n_components=3, shrinkage="auto")
        elif model_name == "csp_logistic":
            return CSPLogRegDecoder(n_components=3, C=1.0, random_state=random_state)
        elif model_name == "eegnet":
            return EEGNetClassifier(epochs=35, batch_size=32, lr=0.005, random_state=random_state)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    def prepare_data(
        self,
        window: Optional[Dict] = None,
        feature_type: str = "combined",
        channels_subset: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Extract multi-channel EEG segments, extracted features, and trial metadata.
        
        Args:
            window: Dict specifying window slicing, or None for full [-1000, 0] ms epoch
            feature_type: 'time_domain', 'bandpower', or 'combined'
            channels_subset: Optional channel list, e.g. MOTOR_FOCUSED_CHANNELS
        Returns:
            X_3d: (n_trials, n_samples, n_channels)
            X_feat: (n_trials, n_features)
            meta_df: DataFrame with trial metadata
        """
        n_trials = len(self.trials)
        all_channels = self.trials[0].channels
        
        if channels_subset:
            ch_indices = [all_channels.index(c) for c in channels_subset if c in all_channels]
        else:
            ch_indices = list(range(len(all_channels)))

        n_channels = len(ch_indices)
        
        if window is not None:
            s_idx = window["start_idx"]
            e_idx = window["end_idx"]
        else:
            s_idx = 0
            e_idx = 200

        n_samples = e_idx - s_idx

        X_3d = np.zeros((n_trials, n_samples, n_channels), dtype=np.float32)
        X_feat_list = []
        meta_rows = []

        for i, tr in enumerate(self.trials):
            seg = tr.eeg_pre[s_idx:e_idx, :][:, ch_indices]
            X_3d[i] = seg
            feat = extract_trial_features(seg, feature_type=feature_type, srate=tr.srate_pre)
            X_feat_list.append(feat)

            meta_rows.append({
                "participant_id": tr.participant_id,
                "run_id": tr.run_id,
                "trial_id": tr.trial_id,
                "target_goal": tr.target_goal,
                "physical_hand_direction": tr.physical_hand_direction,
                "congruency": tr.congruency,
                "label_code": tr.label_code,
            })

        X_feat = np.array(X_feat_list, dtype=np.float32)
        meta_df = pd.DataFrame(meta_rows)
        return X_3d, X_feat, meta_df

    def evaluate_loso(
        self,
        model_name: str = "lda",
        target_type: str = "goal",  # 'goal' or 'motor'
        train_cond: str = "all",     # 'all', 'congruent', 'incongruent'
        test_cond: str = "all",      # 'all', 'congruent', 'incongruent'
        window: Optional[Dict] = None,
        feature_type: str = "combined",
        channels_subset: Optional[List[str]] = None,
    ) -> Tuple[Dict, List[Dict]]:
        """Evaluate a model using Leave-One-Subject-Out cross-validation.
        
        Returns:
            summary_metrics: Dict with aggregate balanced accuracy, F1, AUROC, CI
            oof_predictions: List of dicts for every held-out test trial
        """
        X_3d, X_feat, meta_df = self.prepare_data(
            window=window, feature_type=feature_type, channels_subset=channels_subset
        )

        # Decide whether model consumes 3D raw signals or extracted 2D features
        is_raw_model = model_name in ("csp_lda", "csp_logistic", "eegnet")
        X = X_3d if is_raw_model else X_feat

        # Define targets: 0 = left, 1 = right
        if target_type == "goal":
            y = (meta_df["target_goal"] == "right").astype(int).values
        elif target_type == "motor":
            y = (meta_df["physical_hand_direction"] == "right").astype(int).values
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        subject_metrics = []
        oof_records = []

        window_label = f"{int(window['start_ms'])}to{int(window['end_ms'])}ms" if window else "-1000to0ms"

        for fold_idx, test_sub in enumerate(self.subjects):
            test_mask = meta_df["participant_id"] == test_sub
            train_mask = ~test_mask

            # Filter by condition
            if train_cond != "all":
                train_mask = train_mask & (meta_df["congruency"] == train_cond)
            if test_cond != "all":
                test_mask = test_mask & (meta_df["congruency"] == test_cond)

            X_tr, y_tr = X[train_mask], y[train_mask]
            X_te, y_te = X[test_mask], y[test_mask]
            meta_te = meta_df[test_mask]

            if len(X_tr) == 0 or len(X_te) == 0:
                continue

            # Strict train-fold model fitting
            clf = self._get_model(model_name, random_state=42 + fold_idx)
            clf.fit(X_tr, y_tr)

            probs = clf.predict_proba(X_te)
            preds = clf.predict(X_te)

            # Store out-of-fold trial predictions
            for j, (_, row) in enumerate(meta_te.iterrows()):
                oof_records.append({
                    "participant_id": row["participant_id"],
                    "run_id": row["run_id"],
                    "trial_id": row["trial_id"],
                    "target_type": target_type,
                    "target_label": "right" if y_te[j] == 1 else "left",
                    "condition": row["congruency"],
                    "prediction": "right" if preds[j] == 1 else "left",
                    "probability_left": float(probs[j, 0]),
                    "probability_right": float(probs[j, 1]),
                    "model_name": model_name,
                    "feature_set": "raw" if is_raw_model else feature_type,
                    "fold_id": test_sub,
                    "time_window": window_label,
                })

            # Calculate fold metrics
            bal_acc = balanced_accuracy_score(y_te, preds)
            f1 = f1_score(y_te, preds, zero_division=0)
            try:
                auc = roc_auc_score(y_te, probs[:, 1])
            except ValueError:
                auc = 0.5

            subject_metrics.append({
                "subject": test_sub,
                "n_train": len(X_tr),
                "n_test": len(X_te),
                "balanced_accuracy": bal_acc,
                "f1": f1,
                "auroc": auc,
            })

        # Summary across folds
        bal_accs = [m["balanced_accuracy"] for m in subject_metrics]
        f1s = [m["f1"] for m in subject_metrics]
        aucs = [m["auroc"] for m in subject_metrics]

        mean_acc = float(np.mean(bal_accs)) if bal_accs else 0.5
        sem_acc = float(stats.sem(bal_accs)) if len(bal_accs) > 1 else 0.0

        summary = {
            "model": model_name,
            "target": target_type,
            "train_cond": train_cond,
            "test_cond": test_cond,
            "time_window": window_label,
            "balanced_accuracy_mean": mean_acc,
            "balanced_accuracy_std": float(np.std(bal_accs)) if bal_accs else 0.0,
            "balanced_accuracy_sem": sem_acc,
            "ci_95_low": max(0.0, mean_acc - 1.96 * sem_acc),
            "ci_95_high": min(1.0, mean_acc + 1.96 * sem_acc),
            "f1_mean": float(np.mean(f1s)) if f1s else 0.0,
            "auroc_mean": float(np.mean(aucs)) if aucs else 0.5,
            "per_subject": subject_metrics,
        }
        return summary, oof_records

    def run_critical_experiment_matrix(
        self, model_names: List[str] = ["lda", "logistic_regression", "csp_lda", "eegnet"]
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """Run the complete 10-condition Experiment Matrix from Section 19."""
        conditions = [
            ("goal", "all", "all"),
            ("goal", "congruent", "congruent"),
            ("goal", "incongruent", "incongruent"),
            ("goal", "congruent", "incongruent"),  # Critical Goal Transfer
            ("goal", "incongruent", "congruent"),
            ("motor", "all", "all"),
            ("motor", "congruent", "congruent"),
            ("motor", "incongruent", "incongruent"),
            ("motor", "congruent", "incongruent"),  # Critical Motor Transfer
            ("motor", "incongruent", "congruent"),
        ]

        summary_rows = []
        all_oofs = []

        for m_name in model_names:
            for target, tr_c, te_c in conditions:
                summ, oofs = self.evaluate_loso(
                    model_name=m_name,
                    target_type=target,
                    train_cond=tr_c,
                    test_cond=te_c,
                    window=None,  # Full pre-movement window [-1000, 0] ms
                )
                summary_rows.append({
                    "Target": target.capitalize(),
                    "Train Condition": tr_c.capitalize(),
                    "Test Condition": te_c.capitalize(),
                    "Model": m_name.upper(),
                    "Split": "LOSO",
                    "Balanced Accuracy": summ["balanced_accuracy_mean"],
                    "Std": summ["balanced_accuracy_std"],
                    "SEM": summ["balanced_accuracy_sem"],
                    "95% CI": f"[{summ['ci_95_low']:.3f}, {summ['ci_95_high']:.3f}]",
                    "F1": summ["f1_mean"],
                    "AUROC": summ["auroc_mean"],
                })
                # Keep OOF records for primary conditions
                if tr_c == "all" and te_c == "all":
                    all_oofs.extend(oofs)

        matrix_df = pd.DataFrame(summary_rows)
        return matrix_df, all_oofs

    def run_temporal_sliding_decoding(
        self, model_name: str = "lda"
    ) -> Dict[str, pd.DataFrame]:
        """Compute temporal decoding trajectory across all 17 pre-movement windows."""
        results = {
            "goal_all": [],
            "motor_all": [],
            "goal_transfer": [],   # Congruent -> Incongruent
            "motor_transfer": [],  # Congruent -> Incongruent
        }

        for w in self.sliding_windows:
            # 1. Goal All->All
            s_ga, _ = self.evaluate_loso(model_name, "goal", "all", "all", window=w)
            results["goal_all"].append({
                "window_id": w["window_id"],
                "center_ms": w["center_ms"],
                "balanced_accuracy": s_ga["balanced_accuracy_mean"],
                "sem": s_ga["balanced_accuracy_sem"],
                "ci_low": s_ga["ci_95_low"],
                "ci_high": s_ga["ci_95_high"],
            })

            # 2. Motor All->All
            s_ma, _ = self.evaluate_loso(model_name, "motor", "all", "all", window=w)
            results["motor_all"].append({
                "window_id": w["window_id"],
                "center_ms": w["center_ms"],
                "balanced_accuracy": s_ma["balanced_accuracy_mean"],
                "sem": s_ma["balanced_accuracy_sem"],
                "ci_low": s_ma["ci_95_low"],
                "ci_high": s_ma["ci_95_high"],
            })

            # 3. Goal Transfer (Congruent -> Incongruent)
            s_gt, _ = self.evaluate_loso(model_name, "goal", "congruent", "incongruent", window=w)
            results["goal_transfer"].append({
                "window_id": w["window_id"],
                "center_ms": w["center_ms"],
                "balanced_accuracy": s_gt["balanced_accuracy_mean"],
                "sem": s_gt["balanced_accuracy_sem"],
                "ci_low": s_gt["ci_95_low"],
                "ci_high": s_gt["ci_95_high"],
            })

            # 4. Motor Transfer (Congruent -> Incongruent)
            s_mt, _ = self.evaluate_loso(model_name, "motor", "congruent", "incongruent", window=w)
            results["motor_transfer"].append({
                "window_id": w["window_id"],
                "center_ms": w["center_ms"],
                "balanced_accuracy": s_mt["balanced_accuracy_mean"],
                "sem": s_mt["balanced_accuracy_sem"],
                "ci_low": s_mt["ci_95_low"],
                "ci_high": s_mt["ci_95_high"],
            })

        dfs = {k: pd.DataFrame(v) for k, v in results.items()}
        return dfs

    def determine_neural_commitment_points(
        self, temporal_dfs: Dict[str, pd.DataFrame], chance_level: float = 0.50
    ) -> Dict[str, Dict]:
        """Estimate the Neural Commitment Point (NCP) for Goal and Motor decoders.
        
        Definition: The earliest pre-movement window where balanced accuracy exceeds
        chance + 1.96 * SEM and remains above chance across consecutive windows
        approaching movement onset.
        """
        ncps = {}
        for curve_name in ["goal_all", "motor_all", "goal_transfer", "motor_transfer"]:
            df = temporal_dfs[curve_name]
            ncp_window = None
            
            # Check for stability (>= 2 consecutive windows above chance + SEM)
            for i in range(len(df) - 1):
                cur_row = df.iloc[i]
                next_row = df.iloc[i + 1]
                if cur_row["ci_low"] > chance_level and next_row["ci_low"] > chance_level:
                    ncp_window = cur_row
                    break

            if ncp_window is not None:
                ncps[curve_name] = {
                    "detected": True,
                    "center_ms": float(ncp_window["center_ms"]),
                    "window_id": int(ncp_window["window_id"]),
                    "balanced_accuracy": float(ncp_window["balanced_accuracy"]),
                    "status": f"Statistical commitment onset identified at {int(ncp_window['center_ms'])} ms",
                }
            else:
                # Fallback to single peak or earliest point exceeding chance
                sig_rows = df[df["ci_low"] > chance_level]
                if not sig_rows.empty:
                    first_sig = sig_rows.iloc[0]
                    ncps[curve_name] = {
                        "detected": True,
                        "center_ms": float(first_sig["center_ms"]),
                        "window_id": int(first_sig["window_id"]),
                        "balanced_accuracy": float(first_sig["balanced_accuracy"]),
                        "status": f"First above-chance threshold at {int(first_sig['center_ms'])} ms (transient)",
                    }
                else:
                    ncps[curve_name] = {
                        "detected": False,
                        "center_ms": None,
                        "window_id": None,
                        "balanced_accuracy": None,
                        "status": "No statistically reliable pre-movement commitment point detected prior to t=0",
                    }
        return ncps

    def run_channel_ablation(
        self, model_name: str = "lda"
    ) -> Dict[str, Dict]:
        """Ablation: All 26 scalp channels vs Motor-Focused channels (FC3, FCz, FC4, C3, Cz, C4, CP3, CPz, CP4)."""
        summ_all_goal, _ = self.evaluate_loso(model_name, "goal", "all", "all")
        summ_motor_goal, _ = self.evaluate_loso(
            model_name, "goal", "all", "all", channels_subset=MOTOR_FOCUSED_CHANNELS
        )
        summ_all_motor, _ = self.evaluate_loso(model_name, "motor", "all", "all")
        summ_motor_motor, _ = self.evaluate_loso(
            model_name, "motor", "all", "all", channels_subset=MOTOR_FOCUSED_CHANNELS
        )

        return {
            "goal": {
                "all_channels_acc": summ_all_goal["balanced_accuracy_mean"],
                "motor_channels_acc": summ_motor_goal["balanced_accuracy_mean"],
                "delta": summ_motor_goal["balanced_accuracy_mean"] - summ_all_goal["balanced_accuracy_mean"],
            },
            "motor": {
                "all_channels_acc": summ_all_motor["balanced_accuracy_mean"],
                "motor_channels_acc": summ_motor_motor["balanced_accuracy_mean"],
                "delta": summ_motor_motor["balanced_accuracy_mean"] - summ_all_motor["balanced_accuracy_mean"],
            },
        }

    @staticmethod
    def save_out_of_fold_predictions(
        oof_records: List[Dict], output_path: Path = Path("reports/phase_2/out_of_fold_predictions.csv")
    ) -> Path:
        """Save trial-level out-of-fold predictions to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(oof_records)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} out-of-fold prediction records to {output_path}")
        return output_path
