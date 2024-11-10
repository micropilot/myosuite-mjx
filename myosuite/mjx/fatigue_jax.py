from jax import numpy as jp
import jax
import mujoco

class CumulativeFatigueJAX():
    """
    3CC-r fatigue model for muscles using JAX, adapted for MJX environments.
    Based on the original fatigue model from MyoSuite.
    """

    def __init__(self, mj_model, frame_skip=1, seed=None):
        self._r = 10 * 15  # Recovery time multiplier
        self._F = 0.00912  # Fatigue coefficient
        self._R = 0.1 * 0.00094  # Recovery coefficient
        self._dt = mj_model.opt.timestep * frame_skip  # Adjusted time step based on frame skip
        
        # Determine muscle actuators
        muscle_act_ind = mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self.na = jp.sum(muscle_act_ind)  # Number of muscle actuators

        # Initialize tau activation and deactivation values
        self._tauact = jp.array([mj_model.actuator_dynprm[i][0] for i in range(len(muscle_act_ind)) if muscle_act_ind[i]])
        self._taudeact = jp.array([mj_model.actuator_dynprm[i][1] for i in range(len(muscle_act_ind)) if muscle_act_ind[i]])

        # Initialize muscle states
        self._MA = jp.zeros((self.na,))  # Muscle Active
        self._MR = jp.ones((self.na,))   # Muscle Resting
        self._MF = jp.zeros((self.na,))  # Muscle Fatigue
        self.TL = jp.zeros((self.na,))   # Target Load

        # Random number generator for reset
        self.rng_key = jax.random.PRNGKey(seed) if seed is not None else jax.random.PRNGKey(0)

    def set_FatigueCoefficient(self, F):
        self._F = F
    
    def set_RecoveryCoefficient(self, R):
        self._R = R
    
    def set_RecoveryMultiplier(self, r):
        self._r = r
        
    def compute_act(self, act):
        """Compute the active, resting, and fatigue muscle states based on the target load."""
        self.TL = act  # Target load

        # Calculate effective time constants
        self._LD = 1 / self._tauact * (0.5 + 1.5 * self._MA)
        self._LR = (0.5 + 1.5 * self._MA) / self._taudeact

        # Transfer rate between MR and MA
        C = jp.zeros_like(self._MA)
        C = jax.lax.select(
            (self._MA < self.TL) & (self._MR > (self.TL - self._MA)),
            self._LD * (self.TL - self._MA),
            C
        )
        C = jax.lax.select(
            (self._MA < self.TL) & (self._MR <= (self.TL - self._MA)),
            self._LD * self._MR,
            C
        )
        C = jax.lax.select(self._MA >= self.TL, self._LR * (self.TL - self._MA), C)

        # Recovery rate
        rR = jax.lax.select(self._MA >= self.TL, self._r * self._R, self._R)

        # Clip transfer rate C to maintain state bounds between 0 and 1
        C = jp.clip(C, 
                    jp.maximum(-self._MA / self._dt + self._F * self._MA, (self._MR - 1) / self._dt + rR * self._MF),
                    jp.minimum((1 - self._MA) / self._dt + self._F * self._MA, self._MR / self._dt + rR * self._MF))

        # Update muscle states
        dMA = (C - self._F * self._MA) * self._dt
        dMR = (-C + rR * self._MF) * self._dt
        dMF = (self._F * self._MA - rR * self._MF) * self._dt

        self._MA += dMA
        self._MR += dMR
        self._MF += dMF

        return self._MA, self._MR, self._MF

    def get_effort(self):
        """Calculate effort as the norm between active muscle state and target load."""
        return jp.linalg.norm(self._MA - self.TL)

    def reset(self, fatigue_reset_vec=None, fatigue_reset_random=False):
        """Reset the fatigue model."""
        if fatigue_reset_random:
            non_fatigued_muscles = jax.random.uniform(self.rng_key, (self.na,))
            active_percentage = jax.random.uniform(self.rng_key, (self.na,))
            self._MA = non_fatigued_muscles * active_percentage         # Muscle Active
            self._MR = non_fatigued_muscles * (1 - active_percentage)   # Muscle Resting
            self._MF = 1 - non_fatigued_muscles                         # Muscle Fatigue
        elif fatigue_reset_vec is not None:
            assert len(fatigue_reset_vec) == self.na, f"Invalid fatigue vector length (expected {self.na}, got {len(fatigue_reset_vec)})"
            self._MF = jp.array(fatigue_reset_vec)    # Muscle Fatigue
            self._MR = 1 - self._MF                   # Muscle Resting
            self._MA = jp.zeros((self.na,))           # Muscle Active
        else:
            self._MA = jp.zeros((self.na,))           # Muscle Active
            self._MR = jp.ones((self.na,))            # Muscle Resting
            self._MF = jp.zeros((self.na,))           # Muscle Fatigue

    def seed(self, seed=None):
        """Set random number seed."""
        self.rng_key = jax.random.PRNGKey(seed) if seed is not None else jax.random.PRNGKey(0)
    
    @property
    def MF(self):
        return self._MF
    
    @property
    def MR(self):
        return self._MR
    
    @property
    def MA(self):
        return self._MA
    
    @property
    def F(self):
        return self._F
    
    @property
    def R(self):
        return self._R
    
    @property
    def r(self):
        return self._r
