from typing import Union

from gpflow.models import GPR, SGPR, SVGP

from relaxedgp.models import LSVGP, RSVGP

FullBatchModel = Union[GPR, SGPR]
MinibatchModel = Union[RSVGP, SVGP, LSVGP]
Model = Union[FullBatchModel, MinibatchModel]
