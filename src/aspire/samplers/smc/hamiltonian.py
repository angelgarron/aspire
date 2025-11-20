from functools import partial

import array_api_compat.torch as torch
import numpy as np

from ...samples import SMCSamples
from ...utils import asarray, to_numpy, track_calls
from .base import SMCSampler


class HamiltonianSMC(SMCSampler):
    """Hamiltonian SMC sampler."""

    def __init__(
        self,
        log_likelihood,
        log_prior,
        dims,
        prior_flow,
        xp,
        dtype=None,
        parameters=None,
        preconditioning_transform=None,
        rng: np.random.Generator | None = None,  # New parameter
    ):
        # For torch compatibility, we'll keep the original xp
        super().__init__(
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            dims=dims,
            prior_flow=prior_flow,
            xp=xp,
            dtype=dtype,
            parameters=parameters,
            preconditioning_transform=preconditioning_transform,
        )
        self.key = None
        self.rng = rng or np.random.default_rng()

    def log_prob(self, x, beta=None):
        """Log probability function compatible with torch."""
        # Convert to original xp format for computation
        if hasattr(x, "__array__"):
            x_original = asarray(x, self.xp, requires_grad=True)
        else:
            x_original = x

        # Transform back to parameter space
        x_params, log_abs_det_jacobian = (
            self.preconditioning_transform.inverse(x_original)
        )
        samples = SMCSamples(x_params, xp=self.xp, dtype=self.dtype)

        # Compute log probabilities
        log_q = self.prior_flow.log_prob(samples.x)
        samples.log_q = samples.array_to_namespace(log_q)
        samples.log_prior = samples.array_to_namespace(self.log_prior(samples))
        samples.log_likelihood = samples.array_to_namespace(
            self.log_likelihood(samples)
        )

        # Compute target log probability
        log_prob = samples.log_p_t(
            beta=beta
        ).flatten() + samples.array_to_namespace(log_abs_det_jacobian)

        # Handle NaN values
        log_prob = self.xp.where(
            self.xp.isnan(log_prob), -self.xp.inf, log_prob
        )

        return log_prob

    @track_calls
    def sample(
        self,
        n_samples: int,
        n_steps: int = None,
        min_step: float | None = None,
        max_n_steps: int | None = None,
        adaptive: bool = True,
        target_efficiency: float = 0.5,
        target_efficiency_rate: float = 1.0,
        n_final_samples: int | None = None,
        sampler_kwargs: dict | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.sampler_kwargs = sampler_kwargs or {}
        self.sampler_kwargs.setdefault("n_steps", 5 * self.dims)
        self.sampler_kwargs.setdefault("target_acceptance_rate", 0.234)
        self.sampler_kwargs.setdefault("step_fn", "tpcn")
        self.rng = rng or np.random.default_rng()
        return super().sample(
            n_samples,
            n_steps=n_steps,
            adaptive=adaptive,
            target_efficiency=target_efficiency,
            target_efficiency_rate=target_efficiency_rate,
            n_final_samples=n_final_samples,
            min_step=min_step,
            max_n_steps=max_n_steps,
        )

    def mutate(self, particles, beta, n_steps=None):
        log_prob_fn = partial(self.log_prob, beta=beta)

        # Map to transformed dimension for sampling
        z = self.fit_preconditioning_transform(particles.x)

        # Start hamiltonian
        chain = []

        for i in range(n_steps or self.sampler_kwargs["n_steps"]):
            z = _hmc_step(z, log_prob_fn)
            chain.append(z.clone())
            print(f"Step {i}")
        z_final = chain[-1]
        # End hamiltonian

        # Convert back to parameter space
        z_final_np = to_numpy(z_final)
        x = self.preconditioning_transform.inverse(z_final_np)[0]

        # self.history.mcmc_acceptance.append(np.mean(history.acceptance_rate))

        samples = SMCSamples(x, xp=self.xp, beta=beta, dtype=self.dtype)
        samples.log_q = samples.array_to_namespace(
            self.prior_flow.log_prob(samples.x)
        )
        samples.log_prior = samples.array_to_namespace(self.log_prior(samples))
        samples.log_likelihood = samples.array_to_namespace(
            self.log_likelihood(samples)
        )
        if samples.xp.isnan(samples.log_q).any():
            raise ValueError("Log proposal contains NaN values")
        return samples


def _hmc_step(x, log_prob, step_size=0.05, n_steps=10):
    x = x.clone().detach().requires_grad_(True)

    # Sample momentum
    p = torch.randn_like(x)

    # Hamiltonian at start
    current_log_prob = log_prob(x)
    current_H = -current_log_prob + 0.5 * (p**2).sum(-1)

    # Leapfrog integration
    x_new = x
    p_new = p

    # initial half step for momentum
    grad = torch.autograd.grad(current_log_prob.sum(), x_new)[0]
    p_new = p_new + 0.5 * step_size * grad

    for _ in range(n_steps):
        # full step for position
        x_new = x_new + step_size * p_new

        # compute gradient at new position
        x_new = x_new.clone().detach().requires_grad_(True)
        lp = log_prob(x_new)
        grad = torch.autograd.grad(lp.sum(), x_new)[0]

        # full step for momentum (except last iteration)
        p_new = p_new + step_size * grad

    # final half-step momentum
    p_new = p_new - 0.5 * step_size * grad

    # Negate momentum for symmetry
    p_new = -p_new

    # Hamiltonian at end
    new_log_prob = log_prob(x_new)
    new_H = -new_log_prob + 0.5 * (p_new**2).sum(-1)

    # Metropolis acceptance
    accept_prob = torch.exp(current_H - new_H)
    accept = torch.rand_like(accept_prob) < accept_prob

    x_out = torch.where(accept.unsqueeze(-1), x_new.detach(), x.detach())
    return x_out
