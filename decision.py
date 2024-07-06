import numpy as np
import heapq

# Define a Node class for A* pathfinding
class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to node
        self.h = 0  # Heuristic cost from node to goal
        self.f = 0  # Total cost

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

# A* Pathfinding function
def a_star(start, goal, grid):
    start_node = Node(start)
    goal_node = Node(goal)
    open_list = []
    closed_list = set()
    heapq.heappush(open_list, start_node)
    
    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)
        
        if current_node == goal_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]
        
        (x, y) = current_node.position
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        
        for next_position in neighbors:
            if next_position in closed_list or grid[next_position[0]][next_position[1]] == 1:
                continue
            
            neighbor = Node(next_position, current_node)
            neighbor.g = current_node.g + 1
            neighbor.h = abs(goal_node.position[0] - neighbor.position[0]) + abs(goal_node.position[1] - neighbor.position[1])
            neighbor.f = neighbor.g + neighbor.h
            
            if any(open_node for open_node in open_list if neighbor == open_node and neighbor.g > open_node.g):
                continue
            
            heapq.heappush(open_list, neighbor)
    
    return None  # No path found

# Helper function to get the current grid from the rover's perspective
def get_grid(Rover):
    grid_size = 100  # Example grid size
    grid = np.zeros((grid_size, grid_size))
    
    # Assuming Rover's position is at the center of the grid
    center = grid_size // 2
    
    for angle, dist in zip(Rover.nav_angles, Rover.nav_dists):
        x = int(center + dist * np.cos(angle))
        y = int(center + dist * np.sin(angle))
        if 0 <= x < grid_size and 0 <= y < grid_size:
            grid[x, y] = 1
    
    return grid, (center, center)

# Refactored decision_step function with enhanced state handling and A* pathfinding
def decision_step(Rover):
    grid, start = get_grid(Rover)
    goal = None  # Define goal based on the context, e.g., waypoint or rock position
    
    if Rover.samples_angles is not None:
        goal = (int(Rover.samples_dists[0] * np.cos(Rover.samples_angles[0])), 
                int(Rover.samples_dists[0] * np.sin(Rover.samples_angles[0])))
    
    if goal is not None:
        path = a_star(start, goal, grid)
        if path:
            next_move = path[1]
            Rover.steer = np.arctan2(next_move[1] - start[1], next_move[0] - start[0]) * 180 / np.pi
    
    if Rover.nav_angles is not None:
        if Rover.mode[-1] == 'forward':
            if Rover.samples_angles is not None:
                goal = (int(Rover.samples_dists[0] * np.cos(Rover.samples_angles[0])), 
                        int(Rover.samples_dists[0] * np.sin(Rover.samples_angles[0])))
                path = a_star(start, goal, grid)
                if path:
                    next_move = path[1]
                    Rover.steer = np.arctan2(next_move[1] - start[1], next_move[0] - start[0]) * 180 / np.pi
            if Rover.vel <= 0.1 and Rover.total_time - Rover.stuck_time > 4:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.mode.append('stuck')
                Rover.stuck_time = Rover.total_time
            elif Rover.vel < Rover.max_vel:
                Rover.throttle = Rover.throttle_set
                Rover.brake = 0
            else:
                Rover.throttle = 0
                Rover.brake = 0
            offset = 0.8 * np.std(Rover.nav_angles) if Rover.total_time > 10 else 0
            Rover.steer = np.clip(np.mean((Rover.nav_angles + offset) * 180 / np.pi), -15, 15)
        elif Rover.mode[-1] == 'stuck':
            if Rover.total_time - Rover.stuck_time > 1:
                Rover.throttle = Rover.throttle_set
                Rover.brake = 0
                offset = 0.8 * np.std(Rover.nav_angles) if Rover.total_time > 10 else 0
                Rover.steer = np.clip(np.mean((Rover.nav_angles + offset) * 180 / np.pi), -15, 15)
                Rover.mode.pop()
            else:
                Rover.throttle = 0
                Rover.brake = 0
                Rover.steer = -15
        elif Rover.mode[-1] == 'rock':
            mean = np.mean(Rover.samples_angles * 180 / np.pi)
            if not np.isnan(mean):
                Rover.steer = np.clip(mean, -15, 15)
            else:
                Rover.mode.pop()
            if Rover.total_time - Rover.rock_time > 20 or Rover.near_sample:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
            elif Rover.vel <= 0 and Rover.total_time - Rover.stuck_time > 10:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.mode.append('stuck')
                Rover.stuck_time = Rover.total_time
            else:
                slow_speed = Rover.max_vel / 2
                if Rover.vel < slow_speed:
                    Rover.throttle = 0.2
                    Rover.brake = 0
                else:
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
        elif Rover.mode[-1] == 'stop':
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            elif Rover.vel <= 0.2:
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    Rover.brake = 0
                    Rover.steer = -15
                elif len(Rover.nav_angles) >= Rover.go_forward:
                    Rover.throttle = Rover.throttle_set
                    Rover.brake = 0
                    offset = 12
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180 / np.pi) + offset, -15, 15)
                    Rover.mode.pop()
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
    
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover
