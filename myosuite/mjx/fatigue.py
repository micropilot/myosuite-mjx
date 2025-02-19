import jax.numpy as jp
import jax
import jax.random as jrandom
import mujoco
from typing import Dict, Tuple, Any, NamedTuple
from jax import tree_util
import numpy as np

# Constants for floating-point precision
_FLOAT_EPS = jp.finfo(jp.float32).eps
_EPS4 = _FLOAT_EPS * 4.0

class FatigueState(NamedTuple):
    """Container for fatigue state"""
    MA: jp.ndarray  # Muscle Active
    MR: jp.ndarray  # Muscle Resting
    MF: jp.ndarray  # Muscle Fatigue
    TL: jp.ndarray  # Target Load

class CumulativeFatigue:
    """
    JAX implementation of the 3CC-r muscle fatigue model
    Adapted from https://dl.acm.org/doi/pdf/10.1145/3313831.3376701
    Based on implementation from Aleksi Ikkala and Florian Fischer
    """
    def __init__(self, mj_model, frame_skip=1):
        # Get muscle actuator indices
        muscle_act_ind = mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        # Convert to concrete integer value
        self.na = int(np.sum(muscle_act_ind))  # Use numpy here since we're in __init__
        
        # Parameters (static)
        self.F = jp.array(0.00912, dtype=jp.float32)
        self.R = jp.array(0.1 * 0.00094, dtype=jp.float32)
        self.r = jp.array(10 * 15, dtype=jp.float32)
        self.dt = jp.array(mj_model.opt.timestep * frame_skip, dtype=jp.float32)
        
        # Get muscle parameters
        self.tauact = jp.array([
            mj_model.actuator_dynprm[i][0]
            for i in range(len(muscle_act_ind))
            if muscle_act_ind[i]
        ], dtype=jp.float32)
        self.taudeact = jp.array([
            mj_model.actuator_dynprm[i][1]
            for i in range(len(muscle_act_ind))
            if muscle_act_ind[i]
        ], dtype=jp.float32)
        
        # Initial state
        self.state = FatigueState(
            MA=jp.zeros(self.na, dtype=jp.float32),
            MR=jp.ones(self.na, dtype=jp.float32),
            MF=jp.zeros(self.na, dtype=jp.float32),
            TL=jp.zeros(self.na, dtype=jp.float32)
        )

    def _tree_flatten(self) -> Tuple[Tuple[Any, ...], Dict]:
        """Flatten the class into children and auxiliary data"""
        # Dynamic values (arrays that change during computation)
        children = (
            self.state.MA, self.state.MR, self.state.MF, self.state.TL,
            self.F, self.R, self.r, self.dt,
            self.tauact, self.taudeact
        )
        
        # Static values
        aux_data = {
            'na': self.na
        }
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """Reconstruct class from flattened data"""
        obj = cls.__new__(cls)  # Create new instance without __init__
        
        # Restore dynamic values
        MA, MR, MF, TL, F, R, r, dt, tauact, taudeact = children
        
        # Restore static values
        obj.na = aux_data['na']
        obj.F, obj.R, obj.r, obj.dt = F, R, r, dt
        obj.tauact, obj.taudeact = tauact, taudeact
        obj.state = FatigueState(MA=MA, MR=MR, MF=MF, TL=TL)
        return obj

    @jax.jit
    def compute_act(self, act):
        """
        Compute muscle activation considering fatigue
        
        Args:
            act: Target activation levels
            
        Returns:
            tuple: (MA, MR, MF) states
        """
        # Update target load
        state = self.state._replace(TL=jp.array(act, dtype=jp.float32))
        
        # Calculate effective time constants
        LD = 1 / self.tauact * (0.5 + 1.5 * state.MA)
        LR = (0.5 + 1.5 * state.MA) / self.taudeact
        
        # Calculate C(t) - transfer rate between MR and MA
        C = jp.zeros_like(state.MA)
        
        # Case 1: MA < TL and MR > (TL - MA)
        mask1 = (state.MA < state.TL) & (state.MR > (state.TL - state.MA))
        C = jp.where(mask1, LD * (state.TL - state.MA), C)
        
        # Case 2: MA < TL and MR <= (TL - MA)
        mask2 = (state.MA < state.TL) & (state.MR <= (state.TL - state.MA))
        C = jp.where(mask2, LD * state.MR, C)
        
        # Case 3: MA >= TL
        mask3 = state.MA >= state.TL
        C = jp.where(mask3, LR * (state.TL - state.MA), C)
        
        # Calculate recovery rate
        rR = jp.where(state.MA >= state.TL, 
                     self.r * self.R,
                     self.R)
        
        # Clip C(t) to ensure states remain between 0 and 1
        C_min = jp.maximum(
            -state.MA / self.dt + self.F * state.MA,
            (state.MR - 1) / self.dt + rR * state.MF
        )
        C_max = jp.minimum(
            (1 - state.MA) / self.dt + self.F * state.MA,
            state.MR / self.dt + rR * state.MF
        )
        C = jp.clip(C, C_min, C_max)
        
        # Update states
        dMA = (C - self.F * state.MA) * self.dt
        dMR = (-C + rR * state.MF) * self.dt
        dMF = (self.F * state.MA - rR * state.MF) * self.dt
        
        # Create new state
        self.state = FatigueState(
            MA=state.MA + dMA,
            MR=state.MR + dMR,
            MF=state.MF + dMF,
            TL=state.TL
        )
        
        return self.state.MA, self.state.MR, self.state.MF

    @jax.jit
    def get_effort(self):
        """Calculate effort as norm of difference between actual and target activation"""
        return jp.linalg.norm(self.state.MA - self.state.TL)

    def reset(self, fatigue_reset_vec=None, fatigue_reset_random=False, key=None):
        """Reset fatigue states"""
        if fatigue_reset_random:
            assert key is not None, "Key required for random reset"
            key1, key2 = jrandom.split(key)
            non_fatigued_muscles = jrandom.uniform(key1, (self.na,))
            active_percentage = jrandom.uniform(key2, (self.na,))
            self.state = FatigueState(
                MA=non_fatigued_muscles * active_percentage,
                MR=non_fatigued_muscles * (1 - active_percentage),
                MF=1 - non_fatigued_muscles,
                TL=jp.zeros(self.na, dtype=jp.float32)
            )
        else:
            if fatigue_reset_vec is not None:
                assert len(fatigue_reset_vec) == self.na, \
                    f"Invalid length of fatigue vector (expected {self.na}, got {len(fatigue_reset_vec)})"
                self.state = FatigueState(
                    MA=jp.zeros(self.na, dtype=jp.float32),
                    MR=1 - jp.array(fatigue_reset_vec, dtype=jp.float32),
                    MF=jp.array(fatigue_reset_vec, dtype=jp.float32),
                    TL=jp.zeros(self.na, dtype=jp.float32)
                )
            else:
                self.state = FatigueState(
                    MA=jp.zeros(self.na, dtype=jp.float32),
                    MR=jp.ones(self.na, dtype=jp.float32),
                    MF=jp.zeros(self.na, dtype=jp.float32),
                    TL=jp.zeros(self.na, dtype=jp.float32)
                )

    def set_FatigueCoefficient(self, F):
        """Set Fatigue coefficient"""
        self.F = jp.array(F, dtype=jp.float32)

    def set_RecoveryCoefficient(self, R):
        """Set Recovery coefficient"""
        self.R = jp.array(R, dtype=jp.float32)

    def set_RecoveryMultiplier(self, r):
        """Set Recovery time multiplier"""
        self.r = jp.array(r, dtype=jp.float32)

# Register the class as a PyTree
tree_util.register_pytree_node(
    CumulativeFatigue,
    CumulativeFatigue._tree_flatten,
    CumulativeFatigue._tree_unflatten
)
