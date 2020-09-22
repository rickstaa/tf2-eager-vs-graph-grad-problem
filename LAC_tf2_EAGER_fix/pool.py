from collections import deque
import numpy as np
import tensorflow as tf


class Pool(object):
    """Memory buffer class."""

    def __init__(
        self, s_dim, a_dim, memory_capacity, store_last_n_paths, min_memory_size,
    ):
        """Initialize memory buffer object.

        Args:
            variant (dict): Dictionary containing all the required memory buffer
            parameters.
        """
        self.memory_capacity = memory_capacity
        self.paths = deque(maxlen=store_last_n_paths)
        self.reset()
        self.memory = {
            "s": np.zeros([1, s_dim]),
            "a": np.zeros([1, a_dim]),
            "r": np.zeros([1, 1]),
            "terminal": np.zeros([1, 1]),
            "s_": np.zeros([1, s_dim]),
        }
        self.memory_pointer = 0
        self.min_memory_size = min_memory_size

    def reset(self):
        """Reset memory buffer.
        """
        self.current_path = {
            "s": [],
            "a": [],
            "r": [],
            "terminal": [],
            "s_": [],
        }

    def store(self, s, a, r, terminal, s_):
        """Store experience tuple.

        Args:
            s (numpy.ndarray): State.
            a (numpy.ndarray): Action.
            r (numpy.ndarray): Reward.
            terminal (numpy.ndarray): Whether the terminal state was reached.
            s_ (numpy.ndarray): Next state.

        Returns:
            int: The current memory buffer size.
        """

        # Store experience in memory buffer
        transition = {
            "s": s,
            "a": a,
            "r": np.array([r]),
            "terminal": np.array([terminal]),
            "s_": s_,
        }
        if len(self.current_path["s"]) < 1:
            for key in transition.keys():
                self.current_path[key] = transition[key][np.newaxis, :]
        else:
            for key in transition.keys():
                self.current_path[key] = np.concatenate(
                    (self.current_path[key], transition[key][np.newaxis, :])
                )
        if terminal == 1.0:
            # FIXME: DIFFERENCE WITH SPINNINGUP
            # NOTE: WHY the hell only update when Paths are terminal? Done because
            # evaluation is on path basis?
            for key in self.current_path.keys():
                self.memory[key] = np.concatenate(
                    (self.memory[key], self.current_path[key]), axis=0
                )
            self.paths.appendleft(self.current_path)
            self.reset()
            self.memory_pointer = len(self.memory["s"])

        # Return current memory buffer size
        return self.memory_pointer

    def sample(self, batch_size):
        """Sample from memory buffer.

        Args:
            batch_size (int): The memory buffer sample size.

        Returns:
            numpy.ndarray: The batch of experiences.
        """
        if self.memory_pointer < self.min_memory_size:
            return None
        else:

            # Sample a random batch of experiences
            indices = np.random.choice(
                min(self.memory_pointer, self.memory_capacity) - 1,
                size=batch_size,
                replace=False,
            ) + max(1, 1 + self.memory_pointer - self.memory_capacity,) * np.ones(
                [batch_size], np.int
            )
            batch = {}
            for key in self.memory.keys():
                if "s" in key:
                    sample = tf.convert_to_tensor(
                        self.memory[key][indices], dtype=tf.float32
                    )
                    batch.update({key: sample})
                else:
                    batch.update(
                        {
                            key: tf.convert_to_tensor(
                                self.memory[key][indices], dtype=tf.float32
                            )
                        }
                    )
            return batch
