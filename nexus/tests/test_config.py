"""Tests for configuration management"""

import pytest
import tempfile
import os
from pathlib import Path
import yaml
import json

from nexus.core.config import ConfigManager, ProfileManager, ConfigTemplateManager
from nexus.utils.validation import validate_config


class TestConfigManager:
    """Test ConfigManager class"""
    
    def test_default_config_creation(self):
        """Test creation with default configuration"""
        config = ConfigManager(config_path=None, auto_reload=False)
        
        assert config.get("nexus.version") == "0.1.0"
        assert config.get("capture.backend") == "dxcam"
        assert config.get("api.port") == 8000
    
    def test_config_loading_yaml(self, temp_dir):
        """Test loading configuration from YAML file"""
        config_data = {
            "nexus": {"debug": True},
            "capture": {"fps": 120}
        }
        
        config_path = temp_dir / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config = ConfigManager(config_path, auto_reload=False)
        
        assert config.get("nexus.debug") is True
        assert config.get("capture.fps") == 120
        # Should merge with defaults
        assert config.get("api.port") == 8000
    
    def test_config_loading_json(self, temp_dir):
        """Test loading configuration from JSON file"""
        config_data = {
            "nexus": {"debug": True},
            "capture": {"fps": 90}
        }
        
        config_path = temp_dir / "test_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        config = ConfigManager(config_path, auto_reload=False)
        
        assert config.get("nexus.debug") is True
        assert config.get("capture.fps") == 90
    
    def test_config_get_set(self, config_manager):
        """Test getting and setting configuration values"""
        # Test get with default
        assert config_manager.get("nonexistent.key", "default") == "default"
        
        # Test set and get
        config_manager.set("test.key", "test_value")
        assert config_manager.get("test.key") == "test_value"
        
        # Test nested set and get
        config_manager.set("nested.deep.key", 42)
        assert config_manager.get("nested.deep.key") == 42
    
    def test_config_save_load(self, temp_dir, config_manager):
        """Test saving and loading configuration"""
        # Modify config
        config_manager.set("test.save", "save_value")
        
        # Save to file
        save_path = temp_dir / "saved_config.yaml"
        config_manager.save(save_path)
        
        # Load new instance
        new_config = ConfigManager(save_path, auto_reload=False)
        assert new_config.get("test.save") == "save_value"
    
    def test_config_validation(self, config_manager):
        """Test configuration validation"""
        # Valid configuration should pass
        valid, errors = config_manager.validate()
        assert valid
        assert len(errors) == 0
        
        # Invalid configuration should fail
        config_manager.set("capture.fps", -1)
        valid, errors = config_manager.validate()
        assert not valid
        assert len(errors) > 0
    
    def test_environment_variable_loading(self, config_manager):
        """Test loading from environment variables"""
        # Set test environment variable
        os.environ["NEXUS_CAPTURE_FPS"] = "144"
        
        config_manager.load_from_env()
        assert config_manager.get("capture.fps") == 144
        
        # Cleanup
        del os.environ["NEXUS_CAPTURE_FPS"]
    
    def test_export_env_vars(self, config_manager):
        """Test exporting configuration as environment variables"""
        config_manager.set("test.export", "export_value")
        
        env_vars = config_manager.export_env_vars()
        
        assert "NEXUS_TEST_EXPORT" in env_vars
        assert env_vars["NEXUS_TEST_EXPORT"] == "export_value"


class TestProfileManager:
    """Test ProfileManager class"""
    
    def test_profile_creation(self, temp_dir):
        """Test creating a configuration profile"""
        profiles_dir = temp_dir / "profiles"
        manager = ProfileManager(profiles_dir)
        
        # Create profile
        manager.create_profile("test_profile")
        
        # Check profile exists
        profile_path = profiles_dir / "test_profile.yaml"
        assert profile_path.exists()
        
        # Check profile in list
        profiles = manager.list_profiles()
        assert "test_profile" in profiles
    
    def test_profile_load_delete(self, temp_dir):
        """Test loading and deleting profiles"""
        profiles_dir = temp_dir / "profiles"
        manager = ProfileManager(profiles_dir)
        
        # Create and load profile
        manager.create_profile("test_profile")
        config = manager.load_profile("test_profile")
        
        assert manager.current_profile == "test_profile"
        assert isinstance(config, ConfigManager)
        
        # Delete profile
        manager.delete_profile("test_profile")
        
        profiles = manager.list_profiles()
        assert "test_profile" not in profiles
        assert manager.current_profile is None
    
    def test_profile_info(self, temp_dir):
        """Test getting profile information"""
        profiles_dir = temp_dir / "profiles"
        manager = ProfileManager(profiles_dir)
        
        manager.create_profile("info_test")
        info = manager.get_profile_info("info_test")
        
        assert info["name"] == "info_test"
        assert "path" in info
        assert "size" in info
        assert "modified" in info


class TestConfigTemplateManager:
    """Test ConfigTemplateManager class"""
    
    def test_list_templates(self):
        """Test listing available templates"""
        templates = ConfigTemplateManager.list_templates()
        
        assert "fps_game" in templates
        assert "strategy_game" in templates
        assert "development" in templates
    
    def test_template_info(self):
        """Test getting template information"""
        info = ConfigTemplateManager.get_template_info("fps_game")
        
        assert info["name"] == "fps_game"
        assert "sections" in info
        assert "description" in info
        assert "config" in info
    
    def test_create_from_template(self, temp_dir):
        """Test creating configuration from template"""
        output_path = temp_dir / "template_config.yaml"
        
        config = ConfigTemplateManager.create_from_template("development", output_path)
        
        assert output_path.exists()
        assert config.get("nexus.debug") is True
        assert config.get("logging.level") == "DEBUG"


class TestConfigValidation:
    """Test configuration validation"""
    
    def test_valid_config(self, test_config):
        """Test validation of valid configuration"""
        result = validate_config(test_config)
        
        assert result.valid
        assert len(result.errors) == 0
    
    def test_invalid_config(self):
        """Test validation of invalid configuration"""
        invalid_config = {
            "capture": {"fps": -1},  # Invalid FPS
            "api": {"port": 99999},  # Invalid port
            "vision": {"confidence_threshold": 2.0}  # Invalid threshold
        }
        
        result = validate_config(invalid_config)
        
        assert not result.valid
        assert len(result.errors) > 0
    
    def test_config_auto_fix(self):
        """Test automatic config fixing"""
        incomplete_config = {
            "capture": {"fps": 60}
            # Missing other required fields
        }
        
        result = validate_config(incomplete_config, fix_errors=True)
        
        # Should have warnings for fixed values
        assert len(result.warnings) > 0
        assert len(result.fixed_values) > 0
    
    def test_environment_validation(self):
        """Test environment variable validation"""
        from nexus.utils.validation import validate_environment_vars
        
        # Set valid environment variables
        os.environ["NEXUS_DEBUG"] = "true"
        os.environ["NEXUS_LOG_LEVEL"] = "INFO"
        
        result = validate_environment_vars()
        assert result.valid
        
        # Set invalid environment variable
        os.environ["NEXUS_LOG_LEVEL"] = "INVALID"
        
        result = validate_environment_vars()
        assert not result.valid
        assert len(result.errors) > 0
        
        # Cleanup
        del os.environ["NEXUS_DEBUG"]
        del os.environ["NEXUS_LOG_LEVEL"]


@pytest.mark.asyncio
class TestConfigIntegration:
    """Test configuration integration with other components"""
    
    async def test_config_with_capture_manager(self, config_manager, mock_capture_backend):
        """Test configuration integration with capture manager"""
        from nexus.capture.capture_manager import CaptureManager
        from nexus.capture.base import CaptureBackend
        
        # Configure capture settings
        config_manager.set("capture.backend", "mss")
        config_manager.set("capture.fps", 30)
        
        # This would normally create a real capture manager
        # For testing, we'll verify the config values are accessible
        assert config_manager.get("capture.backend") == "mss"
        assert config_manager.get("capture.fps") == 30
    
    async def test_config_hot_reload(self, temp_dir):
        """Test configuration hot reloading"""
        config_path = temp_dir / "hot_reload_config.yaml"
        
        # Create initial config
        initial_config = {"nexus": {"debug": False}}
        with open(config_path, 'w') as f:
            yaml.dump(initial_config, f)
        
        # Create config manager with hot reload disabled for test control
        config = ConfigManager(config_path, auto_reload=False)
        assert config.get("nexus.debug") is False
        
        # Modify file
        updated_config = {"nexus": {"debug": True}}
        with open(config_path, 'w') as f:
            yaml.dump(updated_config, f)
        
        # Manual reload
        config.reload()
        assert config.get("nexus.debug") is True