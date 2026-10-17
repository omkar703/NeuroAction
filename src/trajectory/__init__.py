"""Phase 3 Kinematics and Trajectory Forecasting Package."""

from src.trajectory.schema import TrajectorySample, ForecastResult
from src.trajectory.segmentation import TrajectorySegmenter
from src.trajectory.features import extract_kinematic_features
from src.trajectory.baselines import PersistenceBaseline, ConstantVelocityBaseline
from src.trajectory.models import RidgeTrajectoryForecaster, MLPTrajectoryForecaster, GRUTrajectoryForecaster
from src.trajectory.evaluation import LOSOTrajectoryEvaluator
from src.trajectory.inference import KinematicTwinForecaster, forecast_trajectory

__all__ = [
    "TrajectorySample",
    "ForecastResult",
    "TrajectorySegmenter",
    "extract_kinematic_features",
    "PersistenceBaseline",
    "ConstantVelocityBaseline",
    "RidgeTrajectoryForecaster",
    "MLPTrajectoryForecaster",
    "GRUTrajectoryForecaster",
    "LOSOTrajectoryEvaluator",
    "KinematicTwinForecaster",
    "forecast_trajectory",
]
