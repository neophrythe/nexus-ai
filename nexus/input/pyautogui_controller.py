import asyncio
from typing import Optional, Tuple
import pyautogui
import structlog

from nexus.input.base import InputController, InputAction, MouseButton

logger = structlog.get_logger()

# Configure PyAutoGUI
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


class PyAutoGUIController(InputController):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize PyAutoGUI controller"""
        try:
            # Test PyAutoGUI
            pyautogui.size()
            self._initialized = True
            logger.info("PyAutoGUI controller initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PyAutoGUI: {e}")
            raise
    
    async def key_press(self, key: str) -> None:
        """Press a key"""
        if not self._initialized:
            await self.initialize()
        
        await self._add_human_delay()
        
        await asyncio.get_event_loop().run_in_executor(
            None, pyautogui.keyDown, key
        )
        
        self._add_to_history(InputAction(
            action_type="key_press",
            data={"key": key}
        ))
    
    async def key_release(self, key: str) -> None:
        """Release a key"""
        if not self._initialized:
            await self.initialize()
        
        await asyncio.get_event_loop().run_in_executor(
            None, pyautogui.keyUp, key
        )
        
        self._add_to_history(InputAction(
            action_type="key_release",
            data={"key": key}
        ))
    
    async def key_tap(self, key: str, duration: Optional[float] = None) -> None:
        """Press and release a key"""
        if not self._initialized:
            await self.initialize()
        
        await self._add_human_delay()
        
        if duration:
            await self.key_press(key)
            await asyncio.sleep(duration)
            await self.key_release(key)
        else:
            await asyncio.get_event_loop().run_in_executor(
                None, pyautogui.press, key
            )
            
            self._add_to_history(InputAction(
                action_type="key_tap",
                data={"key": key},
                duration=duration or 0
            ))
    
    async def type_text(self, text: str, interval: Optional[float] = None) -> None:
        """Type text with optional interval between characters"""
        if not self._initialized:
            await self.initialize()
        
        await self._add_human_delay()
        
        if self.human_like and interval is None:
            import random
            for char in text:
                await asyncio.get_event_loop().run_in_executor(
                    None, pyautogui.write, char
                )
                await asyncio.sleep(random.uniform(0.05, 0.15))
        else:
            interval = interval or 0
            await asyncio.get_event_loop().run_in_executor(
                None, pyautogui.write, text, interval
            )
        
        self._add_to_history(InputAction(
            action_type="type_text",
            data={"text": text, "interval": interval}
        ))
    
    async def mouse_move(self, x: int, y: int, duration: Optional[float] = None) -> None:
        """Move mouse to position"""
        if not self._initialized:
            await self.initialize()
        
        await self._add_human_delay()
        
        if self.human_like and duration is None:
            duration = 0.5
        
        duration = duration or 0
        
        await asyncio.get_event_loop().run_in_executor(
            None, pyautogui.moveTo, x, y, duration
        )
        
        self._add_to_history(InputAction(
            action_type="mouse_move",
            data={"x": x, "y": y},
            duration=duration
        ))
    
    async def mouse_click(self, button: MouseButton = MouseButton.LEFT, 
                         x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Click mouse button"""
        if not self._initialized:
            await self.initialize()
        
        await self._add_human_delay()
        
        button_str = button.value
        
        if x is not None and y is not None:
            await asyncio.get_event_loop().run_in_executor(
                None, pyautogui.click, x, y, button=button_str
            )
        else:
            await asyncio.get_event_loop().run_in_executor(
                None, pyautogui.click, button=button_str
            )
        
        self._add_to_history(InputAction(
            action_type="mouse_click",
            data={"button": button_str, "x": x, "y": y}
        ))
    
    async def mouse_down(self, button: MouseButton = MouseButton.LEFT) -> None:
        """Press mouse button"""
        if not self._initialized:
            await self.initialize()
        
        button_str = button.value
        
        await asyncio.get_event_loop().run_in_executor(
            None, pyautogui.mouseDown, button=button_str
        )
        
        self._add_to_history(InputAction(
            action_type="mouse_down",
            data={"button": button_str}
        ))
    
    async def mouse_up(self, button: MouseButton = MouseButton.LEFT) -> None:
        """Release mouse button"""
        if not self._initialized:
            await self.initialize()
        
        button_str = button.value
        
        await asyncio.get_event_loop().run_in_executor(
            None, pyautogui.mouseUp, button=button_str
        )
        
        self._add_to_history(InputAction(
            action_type="mouse_up",
            data={"button": button_str}
        ))
    
    async def mouse_scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Scroll mouse wheel"""
        if not self._initialized:
            await self.initialize()
        
        await self._add_human_delay()
        
        if x is not None and y is not None:
            await self.mouse_move(x, y)
        
        await asyncio.get_event_loop().run_in_executor(
            None, pyautogui.scroll, clicks
        )
        
        self._add_to_history(InputAction(
            action_type="mouse_scroll",
            data={"clicks": clicks, "x": x, "y": y}
        ))
    
    async def mouse_drag(self, start_x: int, start_y: int, end_x: int, end_y: int,
                        button: MouseButton = MouseButton.LEFT, duration: float = 1.0) -> None:
        """Drag mouse from start to end position"""
        if not self._initialized:
            await self.initialize()
        
        await self._add_human_delay()
        
        button_str = button.value
        
        # Move to start position
        await self.mouse_move(start_x, start_y, duration=0.2)
        
        # Drag to end position
        await asyncio.get_event_loop().run_in_executor(
            None, pyautogui.dragTo, end_x, end_y, duration, button=button_str
        )
        
        self._add_to_history(InputAction(
            action_type="mouse_drag",
            data={
                "start_x": start_x, "start_y": start_y,
                "end_x": end_x, "end_y": end_y,
                "button": button_str
            },
            duration=duration
        ))
    
    def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position"""
        return pyautogui.position()
    
    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions"""
        return pyautogui.size()
    
    async def screenshot(self, region: Optional[Tuple[int, int, int, int]] = None) -> Any:
        """Take a screenshot"""
        return await asyncio.get_event_loop().run_in_executor(
            None, pyautogui.screenshot, region
        )