import time
import math
from collections import deque
import numpy as np
import torch

class LatencyTracker:
    """Tracks recent latencies and provides max/percentile queries."""
    def __init__(self, maxlen: int = 100):
        self._values = deque(maxlen=maxlen)
        self.reset()

    def reset(self) -> None:
        self._values.clear()
        self.max_latency = 0.0

    def add(self, latency: float) -> None:
        val = float(latency)
        if val < 0: return
        self._values.append(val)
        self.max_latency = max(self.max_latency, val)

    def __len__(self) -> int: return len(self._values)
    def max(self) -> float | None: return self.max_latency
    def percentile(self, q: float) -> float | None:
        if not self._values: return 0.0
        q = float(q)
        if q <= 0.0: return min(self._values)
        if q >= 1.0: return self.max_latency
        vals = np.array(list(self._values), dtype=np.float32)
        return float(np.quantile(vals, q))
    def mean(self) -> float:
        if not self._values: return 0.0
        return float(np.mean(self._values))
    def p95(self) -> float | None: return self.percentile(0.95)


class RTCProcessor:
    """Real-Time Chunking guidance for flow-matching policies.

    Ported from LeRobot / Physical Intelligence RTC. Applies prefix guidance
    during the ODE denoising loop so that the START of a newly generated action
    chunk smoothly continues the UNEXECUTED TAIL of the previous chunk.

    Adapted to ManiFlow's time convention (t: 0->1, x: noise->data), where the
    predicted clean action is x1 = x_t + (1 - t) * v_t.

    Reference: https://www.physicalintelligence.company/download/real_time_chunking.pdf
    """

    def __init__(self, execution_horizon: int = 16, max_guidance_weight: float = 5.0,
                 prefix_attention_schedule: str = "exp"):
        self.execution_horizon = execution_horizon
        self.max_guidance_weight = float(max_guidance_weight)
        self.prefix_attention_schedule = prefix_attention_schedule

    # ---- prefix weight schedules (from LeRobot) ----
    def _linweights(self, start, end, total):
        skip = max(total - end, 0)
        n = total - skip - start
        if end <= start or n <= 0:
            return torch.tensor([])
        return torch.linspace(1, 0, n + 2)[1:-1]

    def _add_trailing_zeros(self, w, total, end):
        z = total - end
        if z <= 0:
            return w
        return torch.cat([w, torch.zeros(z)])

    def _add_leading_ones(self, w, start, total):
        o = min(start, total)
        if o <= 0:
            return w
        return torch.cat([torch.ones(o), w])

    def get_prefix_weights(self, start, end, total, device):
        start = min(start, end)
        if self.prefix_attention_schedule == "zeros":
            w = torch.zeros(total); w[:start] = 1.0
        elif self.prefix_attention_schedule == "ones":
            w = torch.ones(total); w[end:] = 0.0
        elif self.prefix_attention_schedule == "linear":
            lw = self._linweights(start, end, total)
            w = self._add_trailing_zeros(lw, total, end)
            w = self._add_leading_ones(w, start, total)
        elif self.prefix_attention_schedule == "exp":
            lw = self._linweights(start, end, total)
            lw = lw * torch.expm1(lw).div(math.e - 1)
            w = self._add_trailing_zeros(lw, total, end)
            w = self._add_leading_ones(w, start, total)
        else:
            raise ValueError(f"Unknown prefix_attention_schedule: {self.prefix_attention_schedule}")
        return w.to(device)

    def guide(self, x_t, t, v_fn, prev_chunk, inference_delay, execution_horizon=None):
        """Compute RTC-guided velocity using closed-form approximation.

        Uses the PI-RTC closed-form correction (no autograd backward pass).
        Approximates dx1/dx_t as identity, which is standard in the literature
        and makes guidance essentially free (~0ms overhead vs ~300ms for autograd).

        Args:
            x_t: (B, T, A) current latent in normalized action space.
            t: scalar float in [0, 1] (ManiFlow time; 0=noise, 1=data).
            v_fn: callable x -> velocity.
            prev_chunk: (B, Tp, A) unexecuted tail of previous chunk (normalized).
            inference_delay: int, steps consumed during inference (hard-bind prefix).
            execution_horizon: int, length over which guidance decays.
        Returns:
            (B, T, A) guided velocity.
        """
        if execution_horizon is None:
            execution_horizon = self.execution_horizon
        B, T, A = x_t.shape

        # Pad previous chunk to (B, T, A)
        if prev_chunk.shape[1] < T or prev_chunk.shape[2] < A:
            padded = torch.zeros(B, T, A, device=x_t.device, dtype=x_t.dtype)
            padded[:, :prev_chunk.shape[1], :prev_chunk.shape[2]] = prev_chunk
            prev_chunk = padded
        else:
            prev_chunk = prev_chunk[:, :T, :A]

        eh = min(execution_horizon, prev_chunk.shape[1])
        weights = self.get_prefix_weights(inference_delay, eh, T, x_t.device)
        weights = weights.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)

        # Forward pass only — no autograd needed
        with torch.no_grad():
            v_t = v_fn(x_t)

        # Predicted clean action at this step
        one_minus_t = max(1.0 - t, 1e-6)  # avoid div-by-zero at t=1
        x1 = x_t + one_minus_t * v_t

        # Closed-form correction: steer x1 toward prev_chunk
        # (PI-RTC approximation: dx1/dx_t ≈ I, so correction on x1 ≈ correction on v)
        correction = (prev_chunk - x1) * weights

        # Guidance weight schedule (tau = t in ManiFlow convention)
        tau = float(t)
        if tau < 1e-6:
            gw = self.max_guidance_weight
        else:
            sq_one_minus = (1.0 - tau) ** 2
            inv_r2 = (sq_one_minus + tau ** 2) / sq_one_minus
            c = min((1.0 - tau) / tau, self.max_guidance_weight)
            gw = min(c * inv_r2, self.max_guidance_weight)

        # Apply correction scaled by guidance weight.
        # Note: gw already produces velocity-space scaling (matching the autograd
        # version where correction was in velocity-space via chain rule). No
        # additional 1/(1-t) conversion needed with identity approximation.
        guided_v = v_t + gw * correction

        return guided_v