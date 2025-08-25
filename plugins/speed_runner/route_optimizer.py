"""
Route optimization module for Speed Runner Plugin.
"""

import numpy as np
import math
import time
import random
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from collections import deque
import heapq
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
import networkx as nx


@dataclass
class Waypoint:
    """Represents a waypoint in a route."""
    x: float
    y: float
    z: float
    name: str = ""
    mandatory: bool = True
    order_fixed: bool = False
    time_bonus: float = 0.0  # Time saved by visiting
    time_cost: float = 0.0   # Time cost to visit
    
    def distance_to(self, other: 'Waypoint') -> float:
        """Calculate distance to another waypoint."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )
    
    def as_tuple(self) -> Tuple[float, float, float]:
        """Get position as tuple."""
        return (self.x, self.y, self.z)


@dataclass
class OptimizedRoute:
    """Represents an optimized route."""
    waypoints: List[Waypoint]
    total_distance: float
    estimated_time: float
    optimization_method: str
    improvement: float  # Percentage improvement
    metadata: Dict[str, Any]


class RouteOptimizer:
    """Optimizes speedrun routes using various algorithms."""
    
    def __init__(self, movement_speed: float = 10.0):
        """
        Initialize route optimizer.
        
        Args:
            movement_speed: Average movement speed (units per second)
        """
        self.movement_speed = movement_speed
        self.distance_cache = {}
        self.optimization_history = []
    
    def optimize(self, waypoints: List[Waypoint], 
                method: str = "nearest_neighbor",
                **kwargs) -> OptimizedRoute:
        """Optimize route using specified method."""
        
        # Separate fixed and flexible waypoints
        fixed_waypoints = [w for w in waypoints if w.order_fixed]
        flexible_waypoints = [w for w in waypoints if not w.order_fixed]
        
        if not flexible_waypoints:
            # No optimization needed
            return self._create_route_result(
                waypoints, method, improvement=0.0
            )
        
        # Calculate original distance
        original_distance = self._calculate_total_distance(waypoints)
        
        # Apply optimization method
        if method == "nearest_neighbor":
            optimized = self._nearest_neighbor(waypoints)
        elif method == "two_opt":
            optimized = self._two_opt(waypoints, **kwargs)
        elif method == "genetic":
            optimized = self._genetic_algorithm(waypoints, **kwargs)
        elif method == "simulated_annealing":
            optimized = self._simulated_annealing(waypoints, **kwargs)
        elif method == "a_star":
            optimized = self._a_star_path(waypoints, **kwargs)
        else:
            optimized = waypoints
        
        # Calculate improvement
        optimized_distance = self._calculate_total_distance(optimized)
        improvement = (original_distance - optimized_distance) / original_distance * 100
        
        return self._create_route_result(
            optimized, method, improvement
        )
    
    def _nearest_neighbor(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """Nearest neighbor heuristic."""
        if len(waypoints) <= 2:
            return waypoints
        
        result = [waypoints[0]]  # Start point
        remaining = waypoints[1:-1].copy()  # Middle points
        
        current = waypoints[0]
        
        while remaining:
            # Find nearest unvisited waypoint
            nearest = min(
                remaining,
                key=lambda w: self._get_distance(current, w)
            )
            
            result.append(nearest)
            remaining.remove(nearest)
            current = nearest
        
        result.append(waypoints[-1])  # End point
        
        return result
    
    def _two_opt(self, waypoints: List[Waypoint], 
                max_iterations: int = 1000) -> List[Waypoint]:
        """2-opt local search optimization."""
        if len(waypoints) <= 3:
            return waypoints
        
        route = waypoints.copy()
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    # Calculate distances
                    current_dist = (
                        self._get_distance(route[i-1], route[i]) +
                        self._get_distance(route[j], route[j+1])
                    )
                    
                    swap_dist = (
                        self._get_distance(route[i-1], route[j]) +
                        self._get_distance(route[i], route[j+1])
                    )
                    
                    # Check if swap improves route
                    if swap_dist < current_dist:
                        # Reverse segment
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
            
            iterations += 1
        
        return route
    
    def _genetic_algorithm(self, waypoints: List[Waypoint],
                          population_size: int = 50,
                          generations: int = 100,
                          mutation_rate: float = 0.1) -> List[Waypoint]:
        """Genetic algorithm optimization."""
        if len(waypoints) <= 3:
            return waypoints
        
        # Fixed start and end
        start = waypoints[0]
        end = waypoints[-1]
        middle = waypoints[1:-1]
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = middle.copy()
            random.shuffle(individual)
            population.append([start] + individual + [end])
        
        # Evolution
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [
                1.0 / self._calculate_total_distance(individual)
                for individual in population
            ]
            
            # Selection and reproduction
            new_population = []
            
            # Keep best individual (elitism)
            best_idx = np.argmax(fitness_scores)
            new_population.append(population[best_idx])
            
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_select(population, fitness_scores)
                parent2 = self._tournament_select(population, fitness_scores)
                
                # Crossover
                child = self._crossover(parent1, parent2)
                
                # Mutation
                if random.random() < mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        # Return best solution
        fitness_scores = [
            1.0 / self._calculate_total_distance(individual)
            for individual in population
        ]
        best_idx = np.argmax(fitness_scores)
        
        return population[best_idx]
    
    def _tournament_select(self, population: List, fitness: List,
                          tournament_size: int = 3) -> List[Waypoint]:
        """Tournament selection for genetic algorithm."""
        tournament_indices = random.sample(
            range(len(population)), 
            min(tournament_size, len(population))
        )
        
        tournament_fitness = [fitness[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: List[Waypoint], 
                  parent2: List[Waypoint]) -> List[Waypoint]:
        """Order crossover (OX) for genetic algorithm."""
        if len(parent1) <= 3:
            return parent1.copy()
        
        # Extract middle sections (excluding fixed start/end)
        p1_middle = parent1[1:-1]
        p2_middle = parent2[1:-1]
        
        # Select crossover points
        size = len(p1_middle)
        cx_point1 = random.randint(0, size - 1)
        cx_point2 = random.randint(cx_point1 + 1, size)
        
        # Create child
        child_middle = [None] * size
        child_middle[cx_point1:cx_point2] = p1_middle[cx_point1:cx_point2]
        
        # Fill remaining positions from parent2
        p2_pointer = 0
        for i in range(size):
            if child_middle[i] is None:
                while p2_middle[p2_pointer] in child_middle:
                    p2_pointer += 1
                child_middle[i] = p2_middle[p2_pointer]
                p2_pointer += 1
        
        return [parent1[0]] + child_middle + [parent1[-1]]
    
    def _mutate(self, route: List[Waypoint]) -> List[Waypoint]:
        """Swap mutation for genetic algorithm."""
        if len(route) <= 3:
            return route
        
        mutated = route.copy()
        
        # Select two random positions (excluding fixed start/end)
        i = random.randint(1, len(route) - 2)
        j = random.randint(1, len(route) - 2)
        
        # Swap
        mutated[i], mutated[j] = mutated[j], mutated[i]
        
        return mutated
    
    def _simulated_annealing(self, waypoints: List[Waypoint],
                            initial_temp: float = 100.0,
                            cooling_rate: float = 0.95,
                            min_temp: float = 0.1) -> List[Waypoint]:
        """Simulated annealing optimization."""
        if len(waypoints) <= 3:
            return waypoints
        
        current_route = waypoints.copy()
        current_distance = self._calculate_total_distance(current_route)
        
        best_route = current_route.copy()
        best_distance = current_distance
        
        temperature = initial_temp
        
        while temperature > min_temp:
            # Generate neighbor solution
            neighbor = self._generate_neighbor(current_route)
            neighbor_distance = self._calculate_total_distance(neighbor)
            
            # Calculate acceptance probability
            delta = neighbor_distance - current_distance
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_route = neighbor
                current_distance = neighbor_distance
                
                # Update best solution
                if current_distance < best_distance:
                    best_route = current_route.copy()
                    best_distance = current_distance
            
            # Cool down
            temperature *= cooling_rate
        
        return best_route
    
    def _generate_neighbor(self, route: List[Waypoint]) -> List[Waypoint]:
        """Generate neighbor solution for simulated annealing."""
        neighbor = route.copy()
        
        if len(route) <= 3:
            return neighbor
        
        # Random swap of two middle waypoints
        i = random.randint(1, len(route) - 2)
        j = random.randint(1, len(route) - 2)
        
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        return neighbor
    
    def _a_star_path(self, waypoints: List[Waypoint],
                    grid_resolution: float = 10.0) -> List[Waypoint]:
        """A* pathfinding between waypoints."""
        if len(waypoints) <= 2:
            return waypoints
        
        # Create graph of waypoints
        G = nx.Graph()
        
        # Add nodes
        for i, wp in enumerate(waypoints):
            G.add_node(i, pos=wp.as_tuple(), waypoint=wp)
        
        # Add edges with distances as weights
        for i in range(len(waypoints)):
            for j in range(i + 1, len(waypoints)):
                distance = self._get_distance(waypoints[i], waypoints[j])
                G.add_edge(i, j, weight=distance)
        
        # Find shortest path visiting all mandatory waypoints
        mandatory_indices = [
            i for i, wp in enumerate(waypoints) if wp.mandatory
        ]
        
        if len(mandatory_indices) <= 2:
            # Simple path
            path = nx.shortest_path(
                G, 0, len(waypoints) - 1, weight='weight'
            )
        else:
            # TSP approximation using Christofides algorithm
            path = self._tsp_approximation(G, mandatory_indices)
        
        # Convert path to waypoints
        return [waypoints[i] for i in path]
    
    def _tsp_approximation(self, G: nx.Graph, 
                          nodes: List[int]) -> List[int]:
        """TSP approximation using nearest neighbor."""
        if not nodes:
            return []
        
        path = [nodes[0]]
        unvisited = set(nodes[1:])
        
        while unvisited:
            current = path[-1]
            nearest = min(
                unvisited,
                key=lambda n: G[current][n]['weight'] if G.has_edge(current, n) else float('inf')
            )
            path.append(nearest)
            unvisited.remove(nearest)
        
        return path
    
    def _get_distance(self, wp1: Waypoint, wp2: Waypoint) -> float:
        """Get cached distance between waypoints."""
        key = (id(wp1), id(wp2))
        
        if key not in self.distance_cache:
            self.distance_cache[key] = wp1.distance_to(wp2)
        
        return self.distance_cache[key]
    
    def _calculate_total_distance(self, waypoints: List[Waypoint]) -> float:
        """Calculate total distance of route."""
        if len(waypoints) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(waypoints) - 1):
            total += self._get_distance(waypoints[i], waypoints[i + 1])
        
        return total
    
    def _create_route_result(self, waypoints: List[Waypoint],
                           method: str, improvement: float) -> OptimizedRoute:
        """Create optimized route result."""
        total_distance = self._calculate_total_distance(waypoints)
        estimated_time = total_distance / self.movement_speed
        
        # Add time bonuses and costs
        for wp in waypoints:
            estimated_time += wp.time_cost - wp.time_bonus
        
        result = OptimizedRoute(
            waypoints=waypoints,
            total_distance=total_distance,
            estimated_time=estimated_time,
            optimization_method=method,
            improvement=improvement,
            metadata={
                'waypoint_count': len(waypoints),
                'movement_speed': self.movement_speed,
                'timestamp': time.time()
            }
        )
        
        self.optimization_history.append(result)
        
        return result
    
    def suggest_shortcuts(self, waypoints: List[Waypoint],
                         max_deviation: float = 50.0) -> List[Waypoint]:
        """Suggest potential shortcuts in route."""
        suggestions = []
        
        if len(waypoints) < 3:
            return suggestions
        
        for i in range(len(waypoints) - 2):
            # Check if skipping waypoint i+1 is beneficial
            direct_distance = self._get_distance(waypoints[i], waypoints[i + 2])
            path_distance = (
                self._get_distance(waypoints[i], waypoints[i + 1]) +
                self._get_distance(waypoints[i + 1], waypoints[i + 2])
            )
            
            # Calculate deviation
            deviation = path_distance - direct_distance
            
            if deviation > max_deviation and not waypoints[i + 1].mandatory:
                # Suggest shortcut
                shortcut = Waypoint(
                    x=(waypoints[i].x + waypoints[i + 2].x) / 2,
                    y=(waypoints[i].y + waypoints[i + 2].y) / 2,
                    z=(waypoints[i].z + waypoints[i + 2].z) / 2,
                    name=f"Shortcut {i+1}",
                    mandatory=False
                )
                suggestions.append(shortcut)
        
        return suggestions
    
    def analyze_route(self, waypoints: List[Waypoint]) -> Dict[str, Any]:
        """Analyze route characteristics."""
        if not waypoints:
            return {}
        
        distances = []
        for i in range(len(waypoints) - 1):
            distances.append(
                self._get_distance(waypoints[i], waypoints[i + 1])
            )
        
        total_distance = sum(distances)
        
        # Calculate statistics
        analysis = {
            'total_distance': total_distance,
            'segment_count': len(distances),
            'average_segment': np.mean(distances) if distances else 0,
            'longest_segment': max(distances) if distances else 0,
            'shortest_segment': min(distances) if distances else 0,
            'std_deviation': np.std(distances) if distances else 0,
            'estimated_time': total_distance / self.movement_speed,
            'mandatory_waypoints': sum(1 for wp in waypoints if wp.mandatory),
            'optional_waypoints': sum(1 for wp in waypoints if not wp.mandatory)
        }
        
        # Identify potential issues
        issues = []
        
        # Check for long segments
        long_threshold = analysis['average_segment'] * 2
        for i, dist in enumerate(distances):
            if dist > long_threshold:
                issues.append({
                    'type': 'long_segment',
                    'index': i,
                    'distance': dist
                })
        
        # Check for backtracking
        for i in range(1, len(waypoints) - 1):
            prev_to_next = self._get_distance(waypoints[i-1], waypoints[i+1])
            actual_path = distances[i-1] + distances[i]
            
            if prev_to_next < actual_path * 0.5:
                issues.append({
                    'type': 'potential_backtrack',
                    'index': i,
                    'inefficiency': actual_path - prev_to_next
                })
        
        analysis['issues'] = issues
        
        return analysis