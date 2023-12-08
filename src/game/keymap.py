import gym
import pygame
from typing import Dict, List, Tuple


def get_keymap_and_action_names(name: str) -> Tuple[Dict[int, int], List[str]]:
    """
    Return the keymap and action names for the given state name.
    Args:
        name (str): ['empty', 'episode_replay', 'atari'], the name of the state.

    Returns:
        Keymap and action names for the given state name.

            - keymap (dict): a dictionary mapping pygame keys to action indices.
            - action_names (list): a list of action names.
    """
    if name == 'empty':
        return EMPTY_KEYMAP, EMPTY_ACTION_NAMES
    if name == 'episode_replay':
        return EPISODE_REPLAY_KEYMAP, EPISODE_REPLAY_ACTION_NAMES
    if name == 'atari':
        return ATARI_KEYMAP, ATARI_ACTION_NAMES

    # ===== Other games =====
    assert name.startswith('atari/'), "Keymap name must be start with 'atari/'"
    env_id = name.split('atari/')[1]

    # gather all available action names in the environment
    action_names = [x.lower() for x in gym.make(env_id).get_action_meanings()]
    keymap = dict()
    for key, value in ATARI_KEYMAP.items():
        # only map keys to actions that are available in the environment
        action_name = ATARI_ACTION_NAMES[value]
        if action_name in action_names:
            keymap[key] = action_names.index(action_name)   # assign key(board) to action index

    return keymap, action_names


ATARI_ACTION_NAMES = ['noop',
                      'fire',
                      'up',
                      'right',
                      'left',
                      'down',
                      'up-right',
                      'up-left',
                      'down-right',
                      'down-left',
                      'up-fire',
                      'right-fire',
                      'left-fire',
                      'down-fire',
                      'up-right-fire',
                      'up-left-fire',
                      'down-right-fire',
                      'down-left-fire']

ATARI_KEYMAP = {pygame.K_SPACE: 1,

                pygame.K_w: 2,
                pygame.K_d: 3,
                pygame.K_a: 4,
                pygame.K_s: 5,

                pygame.K_t: 6,
                pygame.K_r: 7,
                pygame.K_g: 8,
                pygame.K_f: 9,

                pygame.K_UP: 10,
                pygame.K_RIGHT: 11,
                pygame.K_LEFT: 12,
                pygame.K_DOWN: 13,

                pygame.K_u: 14,
                pygame.K_y: 15,
                pygame.K_j: 16,
                pygame.K_h: 17}

EPISODE_REPLAY_ACTION_NAMES = ['noop',
                               'previous',
                               'next',
                               'previous_10',
                               'next_10',
                               'go_to_start',
                               'load_previous',
                               'load_next',
                               'go_to_train_episodes',
                               'go_to_test_episodes',
                               'go_to_imagination_episodes']

EPISODE_REPLAY_KEYMAP = {pygame.K_LEFT: 1,
                         pygame.K_RIGHT: 2,
                         pygame.K_PAGEDOWN: 3,
                         pygame.K_PAGEUP: 4,
                         pygame.K_SPACE: 5,
                         pygame.K_DOWN: 6,
                         pygame.K_UP: 7,
                         pygame.K_t: 8,
                         pygame.K_y: 9,
                         pygame.K_i: 10}

EMPTY_ACTION_NAMES = ['noop']
EMPTY_KEYMAP = dict()
