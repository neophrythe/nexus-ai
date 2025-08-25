"""
Experiment Manager for Multi-run Tracking and Comparison

Manages multiple training experiments, configurations, and results
with automatic versioning and comparison capabilities.
"""

import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
import structlog

logger = structlog.get_logger()


class ExperimentManager:
    """
    Manages experiments with configuration tracking, versioning,
    and comparison capabilities.
    
    Features:
    - Experiment versioning and tagging
    - Configuration management
    - Results tracking and comparison
    - Automatic backup and restore
    - Hyperparameter search integration
    """
    
    def __init__(self, base_dir: str = "./experiments"):
        """
        Initialize experiment manager.
        
        Args:
            base_dir: Base directory for experiments
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file
        self.metadata_file = self.base_dir / "experiments.json"
        self.experiments = self._load_metadata()
        
        # Current experiment
        self.current_experiment = None
        
        logger.info(f"Experiment manager initialized at {self.base_dir}")
    
    def create_experiment(self, 
                         name: str,
                         config: Dict[str, Any],
                         tags: List[str] = None,
                         description: str = "") -> str:
        """
        Create a new experiment.
        
        Args:
            name: Experiment name
            config: Experiment configuration
            tags: Tags for categorization
            description: Experiment description
            
        Returns:
            Experiment ID
        """
        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{name}_{timestamp}"
        
        # Create experiment directory
        exp_dir = self.base_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_file = exp_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Create experiment metadata
        metadata = {
            'id': exp_id,
            'name': name,
            'created': datetime.now().isoformat(),
            'tags': tags or [],
            'description': description,
            'config_hash': self._hash_config(config),
            'status': 'created',
            'metrics': {},
            'best_metrics': {},
            'directory': str(exp_dir)
        }
        
        # Save metadata
        self.experiments[exp_id] = metadata
        self._save_metadata()
        
        # Set as current experiment
        self.current_experiment = exp_id
        
        # Create subdirectories
        (exp_dir / 'checkpoints').mkdir(exist_ok=True)
        (exp_dir / 'logs').mkdir(exist_ok=True)
        (exp_dir / 'visualizations').mkdir(exist_ok=True)
        (exp_dir / 'results').mkdir(exist_ok=True)
        
        logger.info(f"Created experiment: {exp_id}")
        
        return exp_id
    
    def load_experiment(self, exp_id: str) -> Dict[str, Any]:
        """
        Load an experiment.
        
        Args:
            exp_id: Experiment ID
            
        Returns:
            Experiment configuration and metadata
        """
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")
        
        metadata = self.experiments[exp_id]
        exp_dir = Path(metadata['directory'])
        
        # Load configuration
        config_file = exp_dir / "config.yaml"
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load results if available
        results = {}
        results_file = exp_dir / "results" / "final_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
        
        self.current_experiment = exp_id
        
        return {
            'metadata': metadata,
            'config': config,
            'results': results
        }
    
    def update_metrics(self, exp_id: str, metrics: Dict[str, float]):
        """
        Update experiment metrics.
        
        Args:
            exp_id: Experiment ID
            metrics: Metrics to update
        """
        if exp_id not in self.experiments:
            return
        
        # Update latest metrics
        self.experiments[exp_id]['metrics'].update(metrics)
        
        # Update best metrics
        best = self.experiments[exp_id]['best_metrics']
        for key, value in metrics.items():
            if key not in best or value > best[key]:
                best[key] = value
        
        # Update status
        self.experiments[exp_id]['status'] = 'running'
        self.experiments[exp_id]['last_updated'] = datetime.now().isoformat()
        
        self._save_metadata()
    
    def finalize_experiment(self, exp_id: str, results: Dict[str, Any]):
        """
        Finalize an experiment with results.
        
        Args:
            exp_id: Experiment ID
            results: Final results
        """
        if exp_id not in self.experiments:
            return
        
        metadata = self.experiments[exp_id]
        exp_dir = Path(metadata['directory'])
        
        # Save results
        results_file = exp_dir / "results" / "final_results.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Update metadata
        metadata['status'] = 'completed'
        metadata['completed'] = datetime.now().isoformat()
        metadata['final_metrics'] = results.get('metrics', {})
        
        self._save_metadata()
        
        logger.info(f"Finalized experiment: {exp_id}")
    
    def compare_experiments(self, exp_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            exp_ids: List of experiment IDs to compare
            
        Returns:
            Comparison results
        """
        comparison = {
            'experiments': {},
            'metrics_comparison': {},
            'config_differences': {}
        }
        
        configs = []
        
        for exp_id in exp_ids:
            if exp_id not in self.experiments:
                continue
            
            # Load experiment
            exp_data = self.load_experiment(exp_id)
            
            comparison['experiments'][exp_id] = {
                'name': exp_data['metadata']['name'],
                'created': exp_data['metadata']['created'],
                'status': exp_data['metadata']['status'],
                'best_metrics': exp_data['metadata'].get('best_metrics', {}),
                'config': exp_data['config']
            }
            
            configs.append(exp_data['config'])
        
        # Compare metrics
        all_metrics = set()
        for exp_id in exp_ids:
            if exp_id in self.experiments:
                all_metrics.update(self.experiments[exp_id].get('best_metrics', {}).keys())
        
        for metric in all_metrics:
            comparison['metrics_comparison'][metric] = {}
            for exp_id in exp_ids:
                if exp_id in self.experiments:
                    value = self.experiments[exp_id].get('best_metrics', {}).get(metric)
                    comparison['metrics_comparison'][metric][exp_id] = value
        
        # Find config differences
        if configs:
            common_keys = set(configs[0].keys())
            for config in configs[1:]:
                common_keys &= set(config.keys())
            
            for key in common_keys:
                values = [config.get(key) for config in configs]
                if len(set(map(str, values))) > 1:  # Different values
                    comparison['config_differences'][key] = {
                        exp_ids[i]: values[i] for i in range(len(exp_ids))
                    }
        
        return comparison
    
    def find_experiments(self, 
                        tags: List[str] = None,
                        status: str = None,
                        name_pattern: str = None) -> List[str]:
        """
        Find experiments matching criteria.
        
        Args:
            tags: Tags to match (any)
            status: Status to match
            name_pattern: Name pattern to match
            
        Returns:
            List of matching experiment IDs
        """
        matches = []
        
        for exp_id, metadata in self.experiments.items():
            # Check tags
            if tags and not any(tag in metadata.get('tags', []) for tag in tags):
                continue
            
            # Check status
            if status and metadata.get('status') != status:
                continue
            
            # Check name pattern
            if name_pattern and name_pattern not in metadata.get('name', ''):
                continue
            
            matches.append(exp_id)
        
        return sorted(matches)
    
    def get_best_experiment(self, metric: str, minimize: bool = False) -> Optional[str]:
        """
        Get the best experiment based on a metric.
        
        Args:
            metric: Metric to optimize
            minimize: Whether to minimize (False = maximize)
            
        Returns:
            Best experiment ID or None
        """
        best_id = None
        best_value = float('inf') if minimize else float('-inf')
        
        for exp_id, metadata in self.experiments.items():
            if metadata.get('status') != 'completed':
                continue
            
            value = metadata.get('best_metrics', {}).get(metric)
            if value is None:
                continue
            
            if (minimize and value < best_value) or (not minimize and value > best_value):
                best_value = value
                best_id = exp_id
        
        return best_id
    
    def backup_experiment(self, exp_id: str, backup_name: str = None):
        """
        Backup an experiment.
        
        Args:
            exp_id: Experiment ID
            backup_name: Optional backup name
        """
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")
        
        metadata = self.experiments[exp_id]
        exp_dir = Path(metadata['directory'])
        
        # Create backup directory
        backup_dir = self.base_dir / 'backups'
        backup_dir.mkdir(exist_ok=True)
        
        # Generate backup name
        if not backup_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{exp_id}_backup_{timestamp}"
        
        backup_path = backup_dir / backup_name
        
        # Copy experiment directory
        shutil.copytree(exp_dir, backup_path)
        
        logger.info(f"Backed up experiment {exp_id} to {backup_path}")
    
    def delete_experiment(self, exp_id: str, backup: bool = True):
        """
        Delete an experiment.
        
        Args:
            exp_id: Experiment ID
            backup: Whether to backup before deletion
        """
        if exp_id not in self.experiments:
            return
        
        # Backup if requested
        if backup:
            self.backup_experiment(exp_id)
        
        # Delete directory
        metadata = self.experiments[exp_id]
        exp_dir = Path(metadata['directory'])
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
        
        # Remove from metadata
        del self.experiments[exp_id]
        self._save_metadata()
        
        logger.info(f"Deleted experiment: {exp_id}")
    
    def export_experiment(self, exp_id: str, export_path: str):
        """
        Export an experiment for sharing.
        
        Args:
            exp_id: Experiment ID
            export_path: Path to export archive
        """
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")
        
        metadata = self.experiments[exp_id]
        exp_dir = Path(metadata['directory'])
        
        # Create archive
        shutil.make_archive(export_path.replace('.zip', ''), 'zip', exp_dir)
        
        logger.info(f"Exported experiment {exp_id} to {export_path}")
    
    def generate_report(self, exp_id: str) -> str:
        """
        Generate a report for an experiment.
        
        Args:
            exp_id: Experiment ID
            
        Returns:
            Report text
        """
        if exp_id not in self.experiments:
            return f"Experiment {exp_id} not found"
        
        metadata = self.experiments[exp_id]
        exp_data = self.load_experiment(exp_id)
        
        lines = ["=" * 60]
        lines.append(f"EXPERIMENT REPORT: {exp_id}")
        lines.append("=" * 60)
        
        # Metadata
        lines.append("\nMETADATA:")
        lines.append(f"  Name: {metadata['name']}")
        lines.append(f"  Created: {metadata['created']}")
        lines.append(f"  Status: {metadata['status']}")
        lines.append(f"  Tags: {', '.join(metadata.get('tags', []))}")
        lines.append(f"  Description: {metadata.get('description', '')}")
        
        # Configuration
        lines.append("\nCONFIGURATION:")
        config = exp_data['config']
        for key, value in config.items():
            lines.append(f"  {key}: {value}")
        
        # Metrics
        if metadata.get('best_metrics'):
            lines.append("\nBEST METRICS:")
            for key, value in metadata['best_metrics'].items():
                lines.append(f"  {key}: {value:.4f}")
        
        # Results
        if exp_data.get('results'):
            lines.append("\nFINAL RESULTS:")
            for key, value in exp_data['results'].items():
                if isinstance(value, (int, float)):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate hash for configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load experiments metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save experiments metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)