import gym
import numpy as np
import pygame

from datetime import datetime
from pathlib import Path
from PIL import Image
from pygame import Surface, SurfaceType, Rect
from pygame.font import Font
from pygame.rect import RectType
from pygame.time import Clock

from src.envs import WorldModelEnv
from src.game.keymap import get_keymap_and_action_names
from src.utils import make_video

from typing import Union, Tuple


class Game:
    def __init__(self,
                 env: Union[gym.Env, WorldModelEnv],
                 keymap_name: str,
                 size: Tuple[int, int],
                 fps: int,
                 verbose: bool,
                 record: bool):
        self.env = env
        self.height, self.width = size
        self.fps = fps
        self.verbose = verbose
        self.record = record
        self.keymap, self.action_names = get_keymap_and_action_names(keymap_name)
        self.record_dir = Path('media') / 'recordings'

        print('Actions: ')
        for key, idx in self.keymap.items():
            print(f'  {pygame.key.name(key)}: {self.action_names[idx]}')

    def pygame_run(self) -> Tuple[Union[Surface, SurfaceType], Clock, Font, Union[Rect, RectType]]:
        pygame.init()
        header_height = 100 if self.verbose else 0  # header for verbose mode
        font_size = 20

        screen = pygame.display.set_mode(size=(self.width, self.height + header_height))
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, font_size)
        header_rect = pygame.Rect(0, 0, self.width, header_height)

        return screen, clock, font, header_rect

    def game_env_step(self, 
                      screen: Surface,
                      clock: Clock,
                      font: Font,
                      header_rect: Rect) -> None:

        def clear_header():
            """Clear the header."""
            pygame.draw.rect(screen, pygame.Color('black'), header_rect)
            pygame.draw.rect(screen, pygame.Color('white'), header_rect, 1)

        def draw_text(text: str, idx_line: int, idx_column: int = 0):
            """
            Draw text in the header.

            Args:
                text: Text to draw.
                idx_line: Line index.
                idx_column: Column index.
            """
            pos = (5 + idx_column * int(self.width // 4), 5 + idx_line * font.size)
            assert (0 <= pos[0] <= self.width) and (0 <= pos[1] <= header_rect.height), \
                f'Invalid text position: {pos}, position must be within the header: {header_rect}'
            screen.blit(font.render(text, antialias=True, color=pygame.Color('white')), dest=pos)
        
        def draw_game(image: Image.Image):
            """Draw the game screen."""
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                assert isinstance(image, Image.Image)
            pygame_image = np.array(image.resize((self.width, self.height), resample=Image.NEAREST))
            pygame_image = pygame_image.transpose(1, 0, 2)  # (H, W, C) -> (W, H, C)
            surface = pygame.surfarray.make_surface(pygame_image)
            screen.blit(surface, dest=(0, header_rect.height))
        
        if isinstance(self.env, gym.Env):
            _, info = self.env.reset(return_info=True)
            img = info['rgn']
        else:
            self.env.reset()
            img = self.env.render()
        
        draw_game(img)
        clear_header()
        pygame.display.flip()   # update the display

        # ===== Main loop =====

        episode_buffer = []
        segment_buffer = []
        recording = False

        do_reset, do_wait = False, False
        should_stop = False

        while not should_stop:
            action = 0  # no-op, do nothing
            pygame.event.pump()  # process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    should_stop = True
                if event.type == pygame.KEYDOWN:
                    action = self.keymap.get(event.key, action)
                    
                    # === Special keys ===
                    if event.key == pygame.K_RETURN:
                        do_reset = True
                    if event.key == pygame.K_PERIOD:
                        do_wait = not do_wait
                    if event.key == pygame.K_COMMA:
                        if not recording:
                            recording = True
                            print('Recording started.')
                        else:
                            print('Stopped recording.')
                            self.save_recording(np.stack(episode_buffer))
                            recording = False
                            segment_buffer.clear()

            if action == 0:
                pressed = pygame.key.get_pressed()
                for key, action in self.keymap.items():
                    if pressed[key]:
                        break
                else:
                    action = 0  # no-op, do nothing

            if not do_wait:
                next_obs, reward, done, info = self.env.step(action)
                img = info['rgb'] if isinstance(self.env, gym.Env) else self.env.render()
                draw_game(img)

                if recording:
                    segment_buffer.append(np.array(img))
                if self.record:
                    episode_buffer.append(np.array(img))
                if self.verbose:
                    clear_header()
                    draw_text(f'Action: {self.action_names[action]}', idx_line=0)
                    draw_text(f'Reward: {reward if isinstance(reward, float) else reward.iteem(): .2f}', idx_line=1)
                    draw_text(f'Done: {done}', idx_line=2)
                    if info is not None:
                        assert isinstance(info, dict)
                        for i, (k, v) in enumerate(info.items()):
                            draw_text(f'{k}: {v}', idx_line=i, idx_column=1)

                pygame.display.flip()   # update the display
                clock.tick(self.fps)

                if do_reset or done:
                    self.env.reset()
                    do_reset = False
                    if self.record:
                        if input('Save episode? [Y/n] '.lower() != 'n'):
                            self.save_recording(np.stack(episode_buffer))
                        episode_buffer.clear()
        
        pygame.quit()

    def run(self) -> None:
        self.game_env_step(*self.pygame_run())

    def save_recording(self, frames: np.ndarray):
        """
        Make video from frames and save it to disk.
        File name is the current timestamp.
        """
        self.record_dir.mkdir(parents=True, exist_ok=True)
        timestep = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        np.save(file=self.record_dir / timestep, arr=frames)
        
        fname = self.record_dir / f"{timestep}.mp4"
        make_video(frames, fps=15, frames=fname)
        print(f'Recording saved to {fname}')