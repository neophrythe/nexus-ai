"""Web Browser Game Launcher for Nexus Framework"""

import asyncio
import time
from typing import Dict, Any, Optional
from pathlib import Path
import psutil
import structlog

from nexus.launchers.base import GameLauncher
from nexus.core.exceptions import LauncherError

logger = structlog.get_logger()


class BrowserLauncher(GameLauncher):
    """Launch browser-based games"""
    
    BROWSER_PATHS = {
        "chrome": {
            "win32": [
                "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
                "%LOCALAPPDATA%\\Google\\Chrome\\Application\\chrome.exe"
            ],
            "linux": [
                "/usr/bin/google-chrome",
                "/usr/bin/chromium-browser",
                "/usr/bin/chromium",
                "/snap/bin/chromium"
            ],
            "darwin": [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
            ]
        },
        "firefox": {
            "win32": [
                "C:\\Program Files\\Mozilla Firefox\\firefox.exe",
                "C:\\Program Files (x86)\\Mozilla Firefox\\firefox.exe"
            ],
            "linux": [
                "/usr/bin/firefox",
                "/usr/bin/firefox-esr",
                "/snap/bin/firefox"
            ],
            "darwin": [
                "/Applications/Firefox.app/Contents/MacOS/firefox"
            ]
        },
        "edge": {
            "win32": [
                "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
                "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe"
            ],
            "linux": [
                "/usr/bin/microsoft-edge",
                "/usr/bin/microsoft-edge-stable"
            ],
            "darwin": [
                "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
            ]
        }
    }
    
    def __init__(self, game_name: str, game_url: str, browser: str = "chrome", 
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(game_name, config)
        self.game_url = game_url
        self.browser = browser
        self.browser_path = self._find_browser()
        self.profile_path = config.get("profile_path") if config else None
        self.window_size = config.get("window_size", "1920,1080") if config else "1920,1080"
        self.fullscreen = config.get("fullscreen", False) if config else False
        self.incognito = config.get("incognito", True) if config else True
        self.disable_web_security = config.get("disable_web_security", False) if config else False
        self.extensions = config.get("extensions", []) if config else []
        
    def _find_browser(self) -> Optional[str]:
        """Find browser installation"""
        import sys
        import os
        
        platform_paths = self.BROWSER_PATHS.get(self.browser, {}).get(sys.platform, [])
        
        # Check configured path first
        if self.config.get("browser_path"):
            if os.path.exists(self.config["browser_path"]):
                return self.config["browser_path"]
        
        # Search default paths
        for path in platform_paths:
            # Expand environment variables
            expanded_path = os.path.expandvars(path)
            if os.path.exists(expanded_path):
                logger.info(f"Found {self.browser} at: {expanded_path}")
                return expanded_path
        
        # Try to find in PATH
        try:
            import shutil
            browser_executable = shutil.which(self.browser)
            if browser_executable:
                return browser_executable
        except Exception as e:
            logger.warning(f"Failed to find browser {self.browser}: {e}")
        
        logger.warning(f"Browser {self.browser} not found")
        return None
    
    async def launch(self) -> bool:
        """Launch game in browser"""
        if not self.browser_path:
            raise LauncherError("browser", self.game_name, "Browser executable not found")
        
        try:
            # Build browser command
            cmd = [self.browser_path]
            
            # Add browser-specific arguments
            if self.browser == "chrome":
                cmd.extend(self._get_chrome_args())
            elif self.browser == "firefox":
                cmd.extend(self._get_firefox_args())
            elif self.browser == "edge":
                cmd.extend(self._get_edge_args())
            
            # Add game URL
            cmd.append(self.game_url)
            
            logger.info(f"Launching {self.game_name} in {self.browser}: {self.game_url}")
            
            # Launch browser
            import subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for browser to start
            await asyncio.sleep(3)
            
            # Find browser process
            for i in range(30):  # Try for 30 seconds
                if self.is_game_running():
                    self.is_running = True
                    logger.info(f"Browser game {self.game_name} launched successfully")
                    return True
                await asyncio.sleep(1)
            
            logger.error(f"Browser game {self.game_name} failed to launch")
            return False
            
        except Exception as e:
            raise LauncherError("browser", self.game_name, f"Failed to launch: {e}")
    
    def _get_chrome_args(self) -> list:
        """Get Chrome-specific arguments"""
        args = []
        
        # Window size
        if not self.fullscreen:
            args.append(f"--window-size={self.window_size}")
        else:
            args.append("--start-fullscreen")
        
        # Profile
        if self.profile_path:
            args.append(f"--user-data-dir={self.profile_path}")
        
        # Incognito mode
        if self.incognito:
            args.append("--incognito")
        
        # Disable web security (for local game development)
        if self.disable_web_security:
            args.extend([
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor",
                "--allow-running-insecure-content"
            ])
        
        # Gaming optimizations
        args.extend([
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            "--disable-background-networking",
            "--process-per-site",
            "--enable-features=VaapiVideoDecoder",
            "--use-gl=desktop"
        ])
        
        # Extensions
        if self.extensions:
            for ext_path in self.extensions:
                args.append(f"--load-extension={ext_path}")
        
        # Game-specific flags
        game_flags = self.config.get("browser_flags", []) if self.config else []
        args.extend(game_flags)
        
        return args
    
    def _get_firefox_args(self) -> list:
        """Get Firefox-specific arguments"""
        args = []
        
        # New instance
        args.append("--new-instance")
        
        # Profile
        if self.profile_path:
            args.extend(["--profile", self.profile_path])
        
        # Private browsing
        if self.incognito:
            args.append("--private-window")
        
        # Fullscreen
        if self.fullscreen:
            args.append("--kiosk")
        
        # Game-specific flags
        game_flags = self.config.get("browser_flags", []) if self.config else []
        args.extend(game_flags)
        
        return args
    
    def _get_edge_args(self) -> list:
        """Get Edge-specific arguments"""
        # Edge uses similar args to Chrome
        args = self._get_chrome_args()
        
        # Edge-specific modifications
        if "--incognito" in args:
            args.remove("--incognito")
            args.append("--inprivate")
        
        return args
    
    async def terminate(self) -> bool:
        """Terminate the browser game"""
        try:
            if self.pid:
                process = psutil.Process(self.pid)
                
                # Try graceful shutdown first
                process.terminate()
                
                # Wait for graceful termination
                try:
                    process.wait(timeout=10)
                except psutil.TimeoutExpired:
                    # Force kill if needed
                    process.kill()
                
                self.pid = None
                self.is_running = False
                logger.info(f"Browser game {self.game_name} terminated")
                return True
            
            # Try to find and kill browser processes with our URL
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info.get('cmdline'):
                        cmdline = ' '.join(proc.info['cmdline'])
                        if (self.browser in proc.info['name'].lower() and 
                            self.game_url in cmdline):
                            proc.terminate()
                            self.is_running = False
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to terminate browser game: {e}")
            return False
    
    def is_game_running(self) -> bool:
        """Check if browser game is running"""
        # Check by PID if we have it
        if self.pid:
            try:
                process = psutil.Process(self.pid)
                return process.is_running()
            except psutil.NoSuchProcess:
                self.pid = None
        
        # Search for browser process with our URL
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info.get('cmdline'):
                    cmdline = ' '.join(proc.info['cmdline'])
                    if (self.browser in proc.info['name'].lower() and 
                        self.game_url in cmdline):
                        self.pid = proc.info['pid']
                        return True
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return False
    
    def get_window_info(self) -> Optional[Dict[str, Any]]:
        """Get browser window information"""
        if not self.is_game_running():
            return None
        
        try:
            import os
            if os.name == 'nt':  # Windows
                import win32gui
                import win32process
                
                def callback(hwnd, windows):
                    if win32gui.IsWindowVisible(hwnd):
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        if pid == self.pid:
                            window_text = win32gui.GetWindowText(hwnd)
                            # Check if window contains our game URL or title
                            if (self.game_url in window_text or 
                                self.game_name.lower() in window_text.lower()):
                                rect = win32gui.GetWindowRect(hwnd)
                                windows.append({
                                    "hwnd": hwnd,
                                    "title": window_text,
                                    "x": rect[0],
                                    "y": rect[1],
                                    "width": rect[2] - rect[0],
                                    "height": rect[3] - rect[1]
                                })
                
                windows = []
                win32gui.EnumWindows(callback, windows)
                
                if windows:
                    # Return main game window
                    return windows[0]
            
            # Fallback for non-Windows
            return {
                "title": f"{self.game_name} - {self.browser}",
                "pid": self.pid,
                "url": self.game_url
            }
            
        except Exception as e:
            logger.error(f"Failed to get window info: {e}")
            return None
    
    def get_browser_info(self) -> Dict[str, Any]:
        """Get browser game information"""
        return {
            "game_url": self.game_url,
            "browser": self.browser,
            "browser_path": self.browser_path,
            "profile_path": self.profile_path,
            "window_size": self.window_size,
            "fullscreen": self.fullscreen,
            "incognito": self.incognito,
            "extensions": self.extensions
        }
    
    def inject_script(self, script_content: str) -> bool:
        """Inject JavaScript into the game page (requires browser automation)"""
        try:
            # This would require selenium or playwright for actual implementation
            logger.warning("Script injection requires browser automation framework (Selenium/Playwright)")
            return False
        except Exception as e:
            logger.error(f"Failed to inject script: {e}")
            return False
    
    async def capture_console_logs(self) -> list:
        """Capture browser console logs (requires browser automation)"""
        try:
            # This would require selenium or playwright for actual implementation
            logger.warning("Console log capture requires browser automation framework (Selenium/Playwright)")
            return []
        except Exception as e:
            logger.error(f"Failed to capture console logs: {e}")
            return []
    
    def set_user_agent(self, user_agent: str) -> None:
        """Set custom user agent string"""
        browser_flags = self.config.get("browser_flags", []) if self.config else []
        
        # Remove existing user-agent flag
        browser_flags = [flag for flag in browser_flags if not flag.startswith("--user-agent")]
        
        # Add new user agent
        browser_flags.append(f"--user-agent={user_agent}")
        
        if self.config:
            self.config["browser_flags"] = browser_flags
    
    def enable_mobile_emulation(self, device_name: str = "iPhone X") -> None:
        """Enable mobile device emulation (Chrome only)"""
        if self.browser != "chrome":
            logger.warning("Mobile emulation only supported in Chrome")
            return
        
        browser_flags = self.config.get("browser_flags", []) if self.config else []
        
        # Add mobile emulation
        mobile_emulation = f'{{"deviceName": "{device_name}"}}'
        browser_flags.append(f"--enable-mobile-emulation={mobile_emulation}")
        
        if self.config:
            self.config["browser_flags"] = browser_flags


class WebGameEnvironment:
    """Environment wrapper for browser-based games"""
    
    def __init__(self, launcher: BrowserLauncher):
        self.launcher = launcher
        self.automation_driver = None
    
    async def initialize_automation(self, framework: str = "playwright"):
        """Initialize browser automation (Playwright or Selenium)"""
        if framework == "playwright":
            try:
                from playwright.async_api import async_playwright
                self.playwright = await async_playwright().start()
                
                browser_type = getattr(self.playwright, self.launcher.browser)
                self.browser = await browser_type.launch()
                self.page = await self.browser.new_page()
                
                logger.info(f"Playwright automation initialized for {self.launcher.browser}")
                return True
            except ImportError:
                logger.error("Playwright not installed. Run: pip install playwright")
                return False
        
        elif framework == "selenium":
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                
                if self.launcher.browser == "chrome":
                    options = Options()
                    for arg in self.launcher._get_chrome_args():
                        options.add_argument(arg)
                    
                    self.automation_driver = webdriver.Chrome(
                        executable_path=self.launcher.browser_path,
                        options=options
                    )
                
                logger.info(f"Selenium automation initialized for {self.launcher.browser}")
                return True
            except ImportError:
                logger.error("Selenium not installed. Run: pip install selenium")
                return False
        
        return False
    
    async def navigate_to_game(self) -> bool:
        """Navigate to the game URL"""
        if self.page:  # Playwright
            await self.page.goto(self.launcher.game_url)
            return True
        elif self.automation_driver:  # Selenium
            self.automation_driver.get(self.launcher.game_url)
            return True
        return False
    
    async def execute_script(self, script: str) -> Any:
        """Execute JavaScript in the game context"""
        if self.page:  # Playwright
            return await self.page.evaluate(script)
        elif self.automation_driver:  # Selenium
            return self.automation_driver.execute_script(script)
        return None
    
    async def get_game_state(self) -> Dict[str, Any]:
        """Extract game state via JavaScript"""
        script = """
        () => {
            return {
                url: window.location.href,
                title: document.title,
                ready_state: document.readyState,
                viewport: {
                    width: window.innerWidth,
                    height: window.innerHeight
                }
            };
        }
        """
        return await self.execute_script(script)
    
    async def cleanup(self):
        """Cleanup automation resources"""
        if self.page:
            await self.page.close()
        if hasattr(self, 'browser'):
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
        if self.automation_driver:
            self.automation_driver.quit()