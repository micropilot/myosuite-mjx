import jax.numpy as jp
import jax
import jax.random as jrandom
import mujoco

# Constants for floating-point precision
_FLOAT_EPS = jp.finfo(jp.float32).eps
_EPS4 = _FLOAT_EPS * 4.0

class CumulativeFatigue:
    """
    JAX implementation of the 3CC-r muscle fatigue model
    Adapted from https://dl.acm.org/doi/pdf/10.1145/3313831.3376701
    Based on implementation from Aleksi Ikkala and Florian Fischer
    """
    def __init__(self, mj_model, frame_skip=1, key=None):
        # Recovery time multiplier (10x factor to compensate for 0.1 below)
        self._r = 10 * 15  
        
        # Fatigue coefficient (identified for elbow torque)
        self._F = jp.array(0.00912, dtype=jp.float32)
        
        # Recovery coefficient (0.1 factor to get ~1% R/F ratio)
        self._R = jp.array(0.1 * 0.00094, dtype=jp.float32)
        
        # Timestep including frame skip
        self._dt = jp.array(mj_model.opt.timestep * frame_skip, dtype=jp.float32)
        
        # Get muscle actuator indices
        muscle_act_ind = mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self.na = int(jp.sum(muscle_act_ind))  # Number of muscle actuators
        
        # Get activation/deactivation time constants for muscles
        self._tauact = jp.array([
            mj_model.actuator_dynprm[i][0]
            for i in range(len(muscle_act_ind))
            if muscle_act_ind[i]
        ], dtype=jp.float32)
        
        self._taudeact = jp.array([
            mj_model.actuator_dynprm[i][1]
            for i in range(len(muscle_act_ind))
            if muscle_act_ind[i]
        ], dtype=jp.float32)
        
        # Initialize state vectors
        self._MA = jp.zeros(self.na, dtype=jp.float32)  # Muscle Active
        self._MR = jp.ones(self.na, dtype=jp.float32)   # Muscle Resting
        self._MF = jp.zeros(self.na, dtype=jp.float32)  # Muscle Fatigue
        self.TL = jp.zeros(self.na, dtype=jp.float32)   # Target Load
        
        # Initialize RNG
        self.key = key

    def set_FatigueCoefficient(self, F):
        """Set Fatigue coefficient"""
        self._F = jp.array(F, dtype=jp.float32)

    def set_RecoveryCoefficient(self, R):
        """Set Recovery coefficient"""
        self._R = jp.array(R, dtype=jp.float32)

    def set_RecoveryMultiplier(self, r):
        """Set Recovery time multiplier"""
        self._r = jp.array(r, dtype=jp.float32)

    def compute_act(self, act):
        """
        Compute muscle activation considering fatigue
        
        Args:
            act: Target activation levels
            
        Returns:
            tuple: Updated (MA, MR, MF) states
        """
        # Set target load
        self.TL = jp.array(act, dtype=jp.float32)
        
        # Calculate effective time constants
        self._LD = 1 / self._tauact * (0.5 + 1.5 * self._MA)
        self._LR = (0.5 + 1.5 * self._MA) / self._taudeact
        
        # Calculate C(t) - transfer rate between MR and MA
        C = jp.zeros_like(self._MA)
        
        # Case 1: MA < TL and MR > (TL - MA)
        mask1 = (self._MA < self.TL) & (self._MR > (self.TL - self._MA))
        C = jp.where(mask1, self._LD * (self.TL - self._MA), C)
        
        # Case 2: MA < TL and MR <= (TL - MA)
        mask2 = (self._MA < self.TL) & (self._MR <= (self.TL - self._MA))
        C = jp.where(mask2, self._LD * self._MR, C)
        
        # Case 3: MA >= TL
        mask3 = self._MA >= self.TL
        C = jp.where(mask3, self._LR * (self.TL - self._MA), C)
        
        # Calculate recovery rate
        rR = jp.where(self._MA >= self.TL, 
                     self._r * self._R,
                     self._R)
        
        # Clip C(t) to ensure states remain between 0 and 1
        C_min = jp.maximum(
            -self._MA / self._dt + self._F * self._MA,
            (self._MR - 1) / self._dt + rR * self._MF
        )
        C_max = jp.minimum(
            (1 - self._MA) / self._dt + self._F * self._MA,
            self._MR / self._dt + rR * self._MF
        )
        C = jp.clip(C, C_min, C_max)
        
        # Update states
        dMA = (C - self._F * self._MA) * self._dt
        dMR = (-C + rR * self._MF) * self._dt
        dMF = (self._F * self._MA - rR * self._MF) * self._dt
        
        self._MA = self._MA + dMA
        self._MR = self._MR + dMR
        self._MF = self._MF + dMF
        
        return self._MA, self._MR, self._MF

    def get_effort(self):
        """Calculate effort as norm of difference between actual and target activation"""
        return jp.linalg.norm(self._MA - self.TL)

    def reset(self, fatigue_reset_vec=None, fatigue_reset_random=False):
        """
        Reset fatigue states
        
        Args:
            fatigue_reset_vec: Optional initial fatigue values
            fatigue_reset_random: Whether to randomize initial states
        """
        if fatigue_reset_random:
            assert fatigue_reset_vec is None, "Cannot use fatigue_reset_vec if fatigue_reset_random=True"
            self.key, key1, key2 = jrandom.split(self.key, 3)
            non_fatigued_muscles = jrandom.uniform(key1, (self.na,))
            active_percentage = jrandom.uniform(key2, (self.na,))
            self._MA = non_fatigued_muscles * active_percentage
            self._MR = non_fatigued_muscles * (1 - active_percentage)
            self._MF = 1 - non_fatigued_muscles
        else:
            if fatigue_reset_vec is not None:
                assert len(fatigue_reset_vec) == self.na, \
                    f"Invalid length of fatigue vector (expected {self.na}, got {len(fatigue_reset_vec)})"
                self._MF = jp.array(fatigue_reset_vec, dtype=jp.float32)
                self._MR = 1 - self._MF
                self._MA = jp.zeros(self.na, dtype=jp.float32)
            else:
                self._MA = jp.zeros(self.na, dtype=jp.float32)
                self._MR = jp.ones(self.na, dtype=jp.float32)
                self._MF = jp.zeros(self.na, dtype=jp.float32)

    # Properties
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
