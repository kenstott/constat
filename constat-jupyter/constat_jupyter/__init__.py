import polars
polars.Config.set_tbl_hide_column_data_types(True)
polars.Config.set_tbl_rows(100)

from .client import ConstatClient, Session
from .models import SolveResult, Artifact, StepInfo, ConstatError

__all__ = ["ConstatClient", "Session", "SolveResult", "Artifact", "StepInfo", "ConstatError"]
__version__ = "0.1.1"
