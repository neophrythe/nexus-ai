"""Frame Transformation Pipeline Orchestration System for Nexus Framework"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json
from pathlib import Path
import structlog
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from collections import OrderedDict

from nexus.vision.frame_processing import GameFrame
from nexus.vision.frame_transformer import FrameTransformer, TransformStep, TransformType

logger = structlog.get_logger()


class PipelineStage(Enum):
    """Pipeline execution stages"""
    PREPROCESSING = "preprocessing"
    FEATURE_EXTRACTION = "feature_extraction" 
    TRANSFORMATION = "transformation"
    ENHANCEMENT = "enhancement"
    POSTPROCESSING = "postprocessing"
    OUTPUT = "output"


class ExecutionMode(Enum):
    """Pipeline execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ASYNC = "async"
    DISTRIBUTED = "distributed"


@dataclass
class PipelineNode:
    """Single node in transformation pipeline"""
    node_id: str
    name: str
    stage: PipelineStage
    transform_type: Union[TransformType, str]
    parameters: Dict[str, Any]
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    condition: Optional[Callable] = None
    enabled: bool = True
    cache_result: bool = False
    timeout_ms: Optional[int] = None
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node transformation"""
        # Check if node should execute
        if not self.enabled:
            return {}
        
        if self.condition and not self.condition(input_data):
            return {}
        
        # Get input frame
        frame = None
        for input_key in self.inputs:
            if input_key in input_data:
                frame = input_data[input_key]
                break
        
        if frame is None:
            logger.warning(f"No input found for node {self.node_id}")
            return {}
        
        # Apply transformation
        # This would be implemented based on transform_type
        result = {self.outputs[0]: frame} if self.outputs else {}
        
        return result


@dataclass
class PipelineEdge:
    """Edge connecting pipeline nodes"""
    source_node: str
    target_node: str
    data_key: str
    transform: Optional[Callable] = None
    
    def transfer_data(self, data: Any) -> Any:
        """Transfer data through edge with optional transformation"""
        if self.transform:
            return self.transform(data)
        return data


@dataclass
class PipelineResult:
    """Result from pipeline execution"""
    success: bool
    outputs: Dict[str, Any]
    execution_time_ms: float
    node_timings: Dict[str, float]
    errors: List[str]
    metadata: Dict[str, Any]


class FrameTransformationPipeline:
    """Advanced frame transformation pipeline orchestrator"""
    
    def __init__(self, name: str = "default"):
        """
        Initialize transformation pipeline
        
        Args:
            name: Pipeline name
        """
        self.name = name
        self.nodes: OrderedDict[str, PipelineNode] = OrderedDict()
        self.edges: List[PipelineEdge] = []
        self.stages: Dict[PipelineStage, List[str]] = {stage: [] for stage in PipelineStage}
        
        # Execution
        self.execution_mode = ExecutionMode.SEQUENTIAL
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
        # Caching
        self.cache: Dict[str, Any] = {}
        self.cache_enabled = True
        self.max_cache_size = 100
        
        # Statistics
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_processing_time_ms": 0,
            "average_processing_time_ms": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Frame transformer
        self.transformer = FrameTransformer()
        
        logger.info(f"Initialized pipeline: {name}")
    
    def add_node(self, node: PipelineNode) -> bool:
        """
        Add node to pipeline
        
        Args:
            node: Pipeline node
        
        Returns:
            True if node added successfully
        """
        if node.node_id in self.nodes:
            logger.warning(f"Node {node.node_id} already exists")
            return False
        
        self.nodes[node.node_id] = node
        self.stages[node.stage].append(node.node_id)
        
        logger.info(f"Added node {node.node_id} to pipeline")
        return True
    
    def add_edge(self, source: str, target: str, data_key: str = "frame",
                 transform: Optional[Callable] = None) -> bool:
        """
        Add edge between nodes
        
        Args:
            source: Source node ID
            target: Target node ID
            data_key: Data key for transfer
            transform: Optional edge transformation
        
        Returns:
            True if edge added successfully
        """
        if source not in self.nodes or target not in self.nodes:
            logger.warning(f"Invalid edge: {source} -> {target}")
            return False
        
        edge = PipelineEdge(source, target, data_key, transform)
        self.edges.append(edge)
        
        # Update node connections
        self.nodes[source].outputs.append(data_key)
        self.nodes[target].inputs.append(data_key)
        
        logger.info(f"Added edge: {source} -> {target}")
        return True
    
    def remove_node(self, node_id: str) -> bool:
        """Remove node from pipeline"""
        if node_id not in self.nodes:
            return False
        
        # Remove node
        node = self.nodes.pop(node_id)
        
        # Remove from stages
        self.stages[node.stage].remove(node_id)
        
        # Remove related edges
        self.edges = [e for e in self.edges 
                      if e.source_node != node_id and e.target_node != node_id]
        
        logger.info(f"Removed node {node_id}")
        return True
    
    def execute(self, input_frame: Union[np.ndarray, GameFrame],
               execution_mode: Optional[ExecutionMode] = None) -> PipelineResult:
        """
        Execute pipeline on input frame
        
        Args:
            input_frame: Input frame
            execution_mode: Override execution mode
        
        Returns:
            Pipeline execution result
        """
        start_time = time.time()
        mode = execution_mode or self.execution_mode
        
        # Convert to GameFrame if needed
        if isinstance(input_frame, np.ndarray):
            input_frame = GameFrame(input_frame)
        
        # Initialize execution context
        context = {
            "input": input_frame,
            "frame": input_frame,
            "metadata": {}
        }
        
        node_timings = {}
        errors = []
        
        try:
            if mode == ExecutionMode.SEQUENTIAL:
                context = self._execute_sequential(context, node_timings)
            elif mode == ExecutionMode.PARALLEL:
                context = self._execute_parallel(context, node_timings)
            elif mode == ExecutionMode.ASYNC:
                context = asyncio.run(self._execute_async(context, node_timings))
            else:
                errors.append(f"Unsupported execution mode: {mode}")
            
            # Update statistics
            self.stats["total_executions"] += 1
            self.stats["successful_executions"] += 1
            
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            errors.append(str(e))
            self.stats["failed_executions"] += 1
        
        execution_time = (time.time() - start_time) * 1000
        self.stats["total_processing_time_ms"] += execution_time
        self.stats["average_processing_time_ms"] = (
            self.stats["total_processing_time_ms"] / max(1, self.stats["total_executions"])
        )
        
        return PipelineResult(
            success=len(errors) == 0,
            outputs=context,
            execution_time_ms=execution_time,
            node_timings=node_timings,
            errors=errors,
            metadata=context.get("metadata", {})
        )
    
    def _execute_sequential(self, context: Dict[str, Any], 
                          timings: Dict[str, float]) -> Dict[str, Any]:
        """Execute pipeline sequentially"""
        # Execute by stages
        for stage in PipelineStage:
            for node_id in self.stages[stage]:
                node = self.nodes[node_id]
                
                # Check cache
                cache_key = self._get_cache_key(node, context)
                if self.cache_enabled and node.cache_result and cache_key in self.cache:
                    result = self.cache[cache_key]
                    self.stats["cache_hits"] += 1
                else:
                    # Execute node
                    node_start = time.time()
                    result = self._execute_node(node, context)
                    timings[node_id] = (time.time() - node_start) * 1000
                    
                    # Cache result
                    if self.cache_enabled and node.cache_result:
                        self._cache_result(cache_key, result)
                    
                    self.stats["cache_misses"] += 1
                
                # Update context
                context.update(result)
        
        return context
    
    def _execute_parallel(self, context: Dict[str, Any],
                        timings: Dict[str, float]) -> Dict[str, Any]:
        """Execute pipeline with parallel stages"""
        import concurrent.futures
        
        for stage in PipelineStage:
            stage_nodes = [self.nodes[node_id] for node_id in self.stages[stage]]
            
            if not stage_nodes:
                continue
            
            # Execute stage nodes in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}
                
                for node in stage_nodes:
                    future = executor.submit(self._execute_node, node, context.copy())
                    futures[future] = node.node_id
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    node_id = futures[future]
                    try:
                        result = future.result(timeout=5.0)
                        context.update(result)
                        timings[node_id] = 0  # Timing not accurate in parallel
                    except Exception as e:
                        logger.error(f"Node {node_id} execution failed: {e}")
        
        return context
    
    async def _execute_async(self, context: Dict[str, Any],
                           timings: Dict[str, float]) -> Dict[str, Any]:
        """Execute pipeline asynchronously"""
        for stage in PipelineStage:
            stage_nodes = [self.nodes[node_id] for node_id in self.stages[stage]]
            
            if not stage_nodes:
                continue
            
            # Create async tasks for stage
            tasks = []
            for node in stage_nodes:
                task = asyncio.create_task(self._execute_node_async(node, context.copy()))
                tasks.append((task, node.node_id))
            
            # Wait for all tasks
            for task, node_id in tasks:
                try:
                    result = await task
                    context.update(result)
                    timings[node_id] = 0  # Timing not accurate in async
                except Exception as e:
                    logger.error(f"Node {node_id} async execution failed: {e}")
        
        return context
    
    def _execute_node(self, node: PipelineNode, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single node"""
        if not node.enabled:
            return {}
        
        if node.condition and not node.condition(context):
            return {}
        
        # Get input frame
        frame = None
        for input_key in node.inputs:
            if input_key in context:
                frame = context[input_key]
                break
        
        if frame is None and node.inputs:
            frame = context.get("frame")
        
        if frame is None:
            return {}
        
        # Apply transformation based on type
        if isinstance(node.transform_type, TransformType):
            # Use frame transformer
            transform_step = TransformStep(
                transform_type=node.transform_type,
                parameters=node.parameters,
                enabled=True,
                name=node.name
            )
            
            if isinstance(frame, GameFrame):
                frame_data = frame.frame_data
            else:
                frame_data = frame
            
            transformed = self.transformer.apply_transform(frame_data, transform_step)
            
            # Create output
            if isinstance(frame, GameFrame):
                output_frame = GameFrame(transformed, metadata={"node": node.node_id})
            else:
                output_frame = transformed
            
            result = {}
            for output_key in node.outputs:
                result[output_key] = output_frame
            
            return result
        
        else:
            # Custom transformation
            return node.execute(context)
    
    async def _execute_node_async(self, node: PipelineNode, 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._execute_node, node, context)
    
    def _get_cache_key(self, node: PipelineNode, context: Dict[str, Any]) -> str:
        """Generate cache key for node execution"""
        # Simple hash based on node ID and input frame ID
        frame = context.get("frame")
        if frame:
            return f"{node.node_id}_{id(frame)}"
        return f"{node.node_id}_{hash(str(context))}"
    
    def _cache_result(self, key: str, result: Any):
        """Cache execution result"""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[key] = result
    
    def clear_cache(self):
        """Clear execution cache"""
        self.cache.clear()
        logger.info("Pipeline cache cleared")
    
    def save_pipeline(self, path: str):
        """Save pipeline configuration to file"""
        config = {
            "name": self.name,
            "execution_mode": self.execution_mode.value,
            "nodes": [],
            "edges": []
        }
        
        # Save nodes
        for node in self.nodes.values():
            node_config = {
                "node_id": node.node_id,
                "name": node.name,
                "stage": node.stage.value,
                "transform_type": node.transform_type.value if isinstance(node.transform_type, TransformType) else str(node.transform_type),
                "parameters": node.parameters,
                "inputs": node.inputs,
                "outputs": node.outputs,
                "enabled": node.enabled,
                "cache_result": node.cache_result,
                "timeout_ms": node.timeout_ms
            }
            config["nodes"].append(node_config)
        
        # Save edges
        for edge in self.edges:
            edge_config = {
                "source": edge.source_node,
                "target": edge.target_node,
                "data_key": edge.data_key
            }
            config["edges"].append(edge_config)
        
        # Save to file
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
        
        logger.info(f"Pipeline saved to {path}")
    
    def load_pipeline(self, path: str):
        """Load pipeline configuration from file"""
        path_obj = Path(path)
        
        if not path_obj.exists():
            logger.error(f"Pipeline file not found: {path}")
            return False
        
        # Load configuration
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            with open(path, 'r') as f:
                config = json.load(f)
        
        # Clear existing pipeline
        self.nodes.clear()
        self.edges.clear()
        self.stages = {stage: [] for stage in PipelineStage}
        
        # Load pipeline
        self.name = config.get("name", "loaded")
        self.execution_mode = ExecutionMode(config.get("execution_mode", "sequential"))
        
        # Load nodes
        for node_config in config.get("nodes", []):
            # Convert transform type
            transform_type_str = node_config["transform_type"]
            try:
                transform_type = TransformType(transform_type_str)
            except ValueError:
                transform_type = transform_type_str
            
            node = PipelineNode(
                node_id=node_config["node_id"],
                name=node_config["name"],
                stage=PipelineStage(node_config["stage"]),
                transform_type=transform_type,
                parameters=node_config.get("parameters", {}),
                inputs=node_config.get("inputs", []),
                outputs=node_config.get("outputs", []),
                enabled=node_config.get("enabled", True),
                cache_result=node_config.get("cache_result", False),
                timeout_ms=node_config.get("timeout_ms")
            )
            
            self.add_node(node)
        
        # Load edges
        for edge_config in config.get("edges", []):
            self.add_edge(
                edge_config["source"],
                edge_config["target"],
                edge_config.get("data_key", "frame")
            )
        
        logger.info(f"Pipeline loaded from {path}")
        return True
    
    def visualize_pipeline(self) -> str:
        """Generate pipeline visualization (Graphviz DOT format)"""
        dot = ["digraph Pipeline {"]
        dot.append('  rankdir=LR;')
        dot.append('  node [shape=box];')
        
        # Group nodes by stage
        for stage in PipelineStage:
            if self.stages[stage]:
                dot.append(f'  subgraph cluster_{stage.value} {{')
                dot.append(f'    label="{stage.value}";')
                dot.append('    style=filled;')
                dot.append('    fillcolor=lightgray;')
                
                for node_id in self.stages[stage]:
                    node = self.nodes[node_id]
                    color = "green" if node.enabled else "red"
                    dot.append(f'    {node_id} [label="{node.name}", color={color}];')
                
                dot.append('  }')
        
        # Add edges
        for edge in self.edges:
            label = edge.data_key if edge.data_key != "frame" else ""
            dot.append(f'  {edge.source_node} -> {edge.target_node} [label="{label}"];')
        
        dot.append('}')
        
        return '\n'.join(dot)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = self.stats.copy()
        stats["num_nodes"] = len(self.nodes)
        stats["num_edges"] = len(self.edges)
        stats["cache_size"] = len(self.cache)
        
        # Stage statistics
        stage_stats = {}
        for stage in PipelineStage:
            stage_stats[stage.value] = len(self.stages[stage])
        stats["stages"] = stage_stats
        
        return stats
    
    def optimize_pipeline(self):
        """Optimize pipeline execution order"""
        # Topological sort for optimal execution order
        from collections import defaultdict, deque
        
        # Build adjacency list
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        for edge in self.edges:
            graph[edge.source_node].append(edge.target_node)
            in_degree[edge.target_node] += 1
        
        # Initialize queue with nodes having no dependencies
        queue = deque()
        for node_id in self.nodes:
            if in_degree[node_id] == 0:
                queue.append(node_id)
        
        # Perform topological sort
        sorted_order = []
        while queue:
            node_id = queue.popleft()
            sorted_order.append(node_id)
            
            for neighbor in graph[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Reorder nodes
        if len(sorted_order) == len(self.nodes):
            new_nodes = OrderedDict()
            for node_id in sorted_order:
                new_nodes[node_id] = self.nodes[node_id]
            self.nodes = new_nodes
            
            logger.info("Pipeline optimized with topological sort")
        else:
            logger.warning("Pipeline contains cycles, optimization skipped")
    
    def validate_pipeline(self) -> List[str]:
        """Validate pipeline configuration"""
        issues = []
        
        # Check for orphan nodes
        connected_nodes = set()
        for edge in self.edges:
            connected_nodes.add(edge.source_node)
            connected_nodes.add(edge.target_node)
        
        for node_id in self.nodes:
            if node_id not in connected_nodes and len(self.nodes) > 1:
                issues.append(f"Node {node_id} is not connected")
        
        # Check for cycles
        if self._has_cycle():
            issues.append("Pipeline contains cycles")
        
        # Check for missing inputs
        for node in self.nodes.values():
            if node.inputs and not any(e.target_node == node.node_id for e in self.edges):
                issues.append(f"Node {node.node_id} has inputs but no incoming edges")
        
        return issues
    
    def _has_cycle(self) -> bool:
        """Check if pipeline has cycles"""
        from collections import defaultdict
        
        # Build adjacency list
        graph = defaultdict(list)
        for edge in self.edges:
            graph[edge.source_node].append(edge.target_node)
        
        # DFS to detect cycle
        visited = set()
        rec_stack = set()
        
        def has_cycle_util(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if has_cycle_util(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.nodes:
            if node not in visited:
                if has_cycle_util(node):
                    return True
        
        return False