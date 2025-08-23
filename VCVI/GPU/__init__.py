# GPU implementations of VCVI algorithms
# These require CUDA-compatible hardware and GPU PyTorch

from .MFVI import MFVI
from .FCVI import FCVI

from .GVC_factor import GVC_factor
from .GVC_orthod import GVC_orthod
from .KVC_GVC_orthod import KVC_GVC_orthod

from .Blocked_factor import Blocked_factor
from .GVC_orthod_withCM import GVC_orthod_withCM

from .YJ import YJ

from .logh_logitreg_autodiff_GPU import LOGH_LOGITREG
from .logh_correlation_lasso_GPU import LOGH_CORRELATION