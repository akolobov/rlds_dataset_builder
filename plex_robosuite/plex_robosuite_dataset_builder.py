from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import os
import glob
import pathlib
import h5py
import random

IMAGE_SHAPE = (128, 128, 3)
INPUT_PATH = "/mnt/d/PLEX_robosuite/"
TRAIN_FRACTION = 0.9

"""
def read_resize_image(path: str, size: Tuple[int, int]) -> np.array:
    data = tf.io.read_file(path)
    image = tf.image.decode_image(data)
    image = tf.image.resize(image, size, method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return image.numpy()
"""


class PLEXRobosuite(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=IMAGE_SHAPE,
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera (called "agentview" in robosuite, i.e., frontal view) RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=IMAGE_SHAPE,
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Eye-in-hand camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(32,),
                            dtype=np.float64,
                            doc='Robot state, consists of'
                                '[7x sines of robot joint angles,'
                                '7x cosines of joint angles,'
                                '7x joint velocities,'
                                '3x eef XYZ position,'
                                '4x eef orientation quaternion,'
                                '2x gripper state,'
                                '2x gripper velocity].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc="Robot action, consists of [3x XYZ delta, 3x roll-pitch-yaw delta, 1x gripper absolute].",
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Always 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='0 at and only at the goal states, -1 at all other states. NOTE: all demos in this dataset spend the last 10 steps at a goal state, so the only states with reward=0 in an episode are the last 10.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step. NOTE: all demos in this dataset spend the last 10 steps at a goal state and only then terminate, i.e., only the very last step out of 10 steps at the goal has the terminal flag set.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),

                'episode_metadata': tfds.features.FeaturesDict({
                    'episode_id': tfds.features.Text(
                        doc='Episode identifier in the <task>__demo_<N> format.'
                    ),
                }),
            }))


    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        train_paths_and_demo_ids = []
        val_paths_and_demo_ids = []
        paths = glob.glob(os.path.join(INPUT_PATH, *("*")))

        for path in paths:
            path = pathlib.Path(path)
            if not path.is_dir():
                continue
            task_name = path.name
            file_path = path / 'Panda' / 'ph' / f'{task_name}.hdf5'
            f = h5py.File(file_path, "r")
            demo_ids = list(f["data"].keys())
            f.close()
            random.shuffle(demo_ids)
            train_paths_and_demo_ids.extend([(task_name, demo_id) for demo_id in demo_ids[:int(len(demo_ids) * TRAIN_FRACTION)]])
            val_paths_and_demo_ids.extend([(task_name, demo_id) for demo_id in demo_ids[int(len(demo_ids) * TRAIN_FRACTION):]])

        return {
            'train': self._generate_examples(path=train_paths_and_demo_ids),
            'val': self._generate_examples(path=val_paths_and_demo_ids),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data
            task_name, demo_id = episode_path.split('__')
            file_path = pathlib.Path(INPUT_PATH) / task_name / 'Panda' / 'ph' / f'{task_name}.hdf5'
            f = h5py.File(file_path, "r")
            demo = {}
            demo['images'] = f[f"data/{demo_id}/obs/agentview_image"][()]
            demo['wrist_images'] = f[f"data/{demo_id}/obs/robot0_eye_in_hand_image"][()]
            demo['proprios'] = f[f"data/{demo_id}/obs/robot0_proprio-state"][()]
            demo["actions"] = f[f"data/{demo_id}/actions"][()]
            demo["successes"] = f[f"data/{demo_id}/successes"][()]
            demo["rewards"] = [(0 if success else -1) for success in demo["successes"]]

            language_instruction = None
            if task_name == "Door":
                language_instruction = "Open the door by pressing down and pulling on the handle."
            elif task_name == "NutAssemblyRound":
                language_instruction = "Put the loop with the handle onto the peg."
            elif task_name == "PickPlaceBread":
                language_instruction = "Place the loaf of bread into the appropriate compartment."
            elif task_name == "PickPlaceMilk":
                language_instruction = "Place the milk carton into the appropriate compartment."
            elif task_name == "PickPlaceCereal":
                language_instruction = "Place the box of cereal into the appropriate compartment."
            elif task_name == "Stack":
                language_instruction = "Place the red cube on top of the green cube."
            else:
                assert not "Known task"
            f.close()

            episode = []
            for i in range(len(demo['images'])):
                # compute Kona language embedding
                language_embedding = self._embed([language_instruction])[0].numpy()

                episode.append({
                    'observation': {
                        'image': demo['images'][i],
                        'wrist_image': demo['wrist_images'][i],
                        'state': demo['proprios'][i],
                    },
                    'action': demo["actions"][i],
                    'discount': 1.0,
                    'reward': float(demo["rewards"][i]),
                    'is_first': i == 0,
                    'is_last': i == (len(demo['images']) - 1),
                    'is_terminal': i == (len(demo['images']) - 1),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'episode_id': episode_path[0] + '__' + episode_path[1]
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = path

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            sample = '__'.join(sample)
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )
