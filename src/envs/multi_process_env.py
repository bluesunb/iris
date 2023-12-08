import gym
import numpy as np

from enum import Enum
from dataclasses import astuple, dataclass
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Any, Callable, List, Optional, Tuple, Union, Iterable

from src.envs.done_tracker import DoneTrackerEnv


class MessageType(Enum):
    """
    Message types for the communication between the parent and child processes.
    """
    RESET = 0
    RESET_RETURN = 1
    STEP = 2
    STEP_RETURN = 3
    CLOSE = 4


@dataclass
class Message:
    """
    Message for the communication between the parent and child processes.
    """
    type: MessageType
    content: Optional[Any] = None

    def __iter__(self) -> Iterable:
        return iter(astuple(self))
    

def child_env(child_id: int, env_fn: Callable[[], gym.Env], child_conn: Connection) -> None:
    np.random.seed(child_id + np.random.randint(0, 2 ** 31 - 1))
    env = env_fn()
    
    while True:
        message_type, content = child_conn.recv()
        if message_type == MessageType.RESET:
            obs = env.reset()
            child_conn.send(Message(MessageType.RESET_RETURN, obs))
        elif message_type == MessageType.STEP:
            obs, reward, done, info = env.step(content)
            if done:
                obs = env.reset()
            child_conn.send(Message(MessageType.STEP_RETURN, (obs, reward, done, info)))
        elif message_type == MessageType.CLOSE:
            child_conn.close()
            return
        else:
            raise NotImplementedError
        

class MultiProcessEnv(DoneTrackerEnv):
    def __init__(self, env_fn: Callable[[], gym.Env], num_envs: int, done_ratio_limit: float):
        """
        Args:
            env_fn: Function that returns a gym.Env instance.
            num_envs: Number of environments.
            wait_envs_ratio: Ratio of environments that should be done before resetting.
        """
        super().__init__(num_envs=num_envs)
        self.num_actions = env_fn().action_space.n
        self.done_ratio_limit = done_ratio_limit
        self.process: List[Process] = []
        self.parent_conns: List[Connection] = []
        
        for child_id in range(self.num_envs):
            parent_conn, child_conn = Pipe()
            self.parent_conns.append(parent_conn)
            proc = Process(target=child_env,
                           args=(child_id, env_fn, child_conn),
                           daemon=True)
            self.process.append(proc)

        for proc in self.process:
            proc.start()

    def should_reset(self) -> bool:
        return (self.num_envs_done / self.num_envs) >= self.done_ratio_limit
    
    def reset(self) -> np.ndarray:
        self.reset_done_tracker()
        for parent_conn in self.parent_conns:
            parent_conn.send(Message(MessageType.RESET))
        
        content = self._recieve(check_type=MessageType.RESET_RETURN)
        return np.stack(content)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        for parent_conn, action in zip(self.parent_conns, actions):
            parent_conn.send(Message(MessageType.STEP, action))
        
        content = self._recieve(check_type=MessageType.STEP_RETURN)
        obs, reward, done, info = zip(*content)
        self.update_done_tracker(done)
        return np.stack(obs), np.stack(reward), done, info

    def _recieve(self, check_type: Optional[MessageType] = None) -> List[Any]:
        """
        Recieve messages from the child processes and return the message contents.

        Args:
            check_type: type of the message desired to recieve.
        Returns:
            List of message contents.
        """
        # recieve messages from the child processes
        messages = [parent_conn.recv() for parent_conn in self.parent_conns]
        if check_type is not None:
            assert all([msg.type == check_type for msg in messages]), \
                f"Expected message type {check_type}, got {messages}"
        return [msg.content for msg in messages]
    
    def close(self) -> None:
        for parent_conn in self.parent_conns:
            parent_conn.send(Message(MessageType.CLOSE))
        for proc in self.process:
            proc.join()
        for parent_conn in self.parent_conns:
            parent_conn.close()