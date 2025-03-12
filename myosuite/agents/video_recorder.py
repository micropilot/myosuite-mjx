import os
import os.path
import wandb
from typing import Callable, Union, List

import numpy as np
from gymnasium import error, logger

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv


class VecVideoRecorder(VecEnvWrapper):
    """
    Wraps a VecEnv or VecEnvWrapper object to record rendered image as mp4 video.
    It requires ffmpeg or avconv to be installed on the machine.

    Note: for now it only allows to record one video and all videos
    must have at least two frames.

    The video recorder code was adapted from Gymnasium v1.0.

    :param venv:
    :param video_folder: Where to save videos
    :param record_video_trigger: Function that defines when to start recording.
                                        The function takes the current number of step,
                                        and returns whether we should start recording or not.
    :param video_length:  Length of recorded videos
    :param name_prefix: Prefix to the video name
    """

    def __init__(
        self,
        venv: VecEnv,
        video_folder: str,
        record_video_trigger: Callable[[int], bool],
        video_length: int = 200,
        name_prefix: str = "rl-video",
        camera_id: Union[int, List[int]] = 0,
        width: int = 640,
        height: int = 480
    ):
        VecEnvWrapper.__init__(self, venv)

        self.env = venv
        # Temp variable to retrieve metadata
        temp_env = venv
        self.camera_id = camera_id  
        self.width = width
        self.height = height
        # Unwrap to retrieve metadata dict
        # that will be used by gym recorder
        while isinstance(temp_env, VecEnvWrapper):
            temp_env = temp_env.venv

        if isinstance(temp_env, DummyVecEnv) or isinstance(temp_env, SubprocVecEnv):
            metadata = temp_env.get_attr("metadata")[0]
        else:
            metadata = temp_env.metadata

        self.env.metadata = metadata
        self.render_mode = "rgb_array"
        # assert self.env.render_mode == "rgb_array", f"The render_mode must be 'rgb_array', not {self.env.render_mode}"

        self.frames_per_sec = self.env.metadata.get("render_fps", 30)

        self.record_video_trigger = record_video_trigger
        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.video_name = f"{self.name_prefix}-step-{self.step_id}-to-step-{self.step_id + self.video_length}.mp4"
        self.video_path = os.path.join(self.video_folder, self.video_name)

        self.recording = False
        self.recorded_frames: list[np.ndarray] = []

        try:
            import moviepy  # noqa: F401
        except ImportError as e:
            raise error.DependencyNotInstalled("MoviePy is not installed, run `pip install 'gymnasium[other]'`") from e

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        if self._video_enabled():
            self._start_video_recorder()
        return obs

    def _start_video_recorder(self) -> None:
        self._start_recording()
        self._capture_frame()

    def _video_enabled(self) -> bool:
        return self.record_video_trigger(self.step_id)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()

        self.step_id += 1
        if self.recording:
            self._capture_frame()
            if len(self.recorded_frames) > self.video_length:
                print(f"Saving video to {self.video_path}")
                self._stop_recording()
        elif self._video_enabled():
            self._start_video_recorder()

        return obs, rewards, dones, infos

    def _capture_frame(self) -> None:
        # Iterate over the list of environments in DummyVecEnv
        for env_idx, env in enumerate(self.env.envs):
            if hasattr(env, "sim") and hasattr(env.sim, "renderer"):
                # Use render_offscreen for myosuite environments
                frame = env.sim.renderer.render_offscreen(width=self.width, 
                                                          height=self.height, 
                                                          camera_id=self.camera_id)
                if isinstance(frame, np.ndarray):
                    self.recorded_frames.append(frame)
                else:
                    print(f"Unexpected frame type: {type(frame)}")
            else:
                raise ValueError(f"Environment at index {env_idx} does not support 'sim.renderer'.")
            break


    def close(self) -> None:
        """Closes the wrapper then the video recorder."""
        VecEnvWrapper.close(self)
        if self.recording:
            self._stop_recording()

    def _start_recording(self) -> None:
        """Start a new recording. If it is already recording, stops the current recording before starting the new one."""
        if self.recording:
            self._stop_recording()

        self.recording = True

    def _stop_recording(self) -> None:
        """Stop current recording and saves the video."""
        assert self.recording, "_stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            clip.write_videofile(self.video_path)
            wandb.log({"evaluation_video": wandb.Video(self.video_path, format="mp4")})

        self.recorded_frames = []
        self.recording = False

    def __del__(self) -> None:
        """Warn the user in case last video wasn't saved."""
        if len(self.recorded_frames) > 0:
            logger.warn("Unable to save last video! Did you call close()?")