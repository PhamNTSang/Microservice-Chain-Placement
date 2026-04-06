import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Optional

from .topology import (
    NetworkTopology, 
    ServiceChain, 
    Microservice,
    create_sample_topology,
    create_sample_service_chain
)

class K8sPlacementEnv(gym.Env):
    """Kubernetes Service Placement Environment."""
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        num_nodes: int = 5,
        num_services: int = 5,
        render_mode: Optional[str] = None,
        failure_prob: float = 0.05  # 5% tỷ lệ hỏng node mỗi step
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_services = num_services
        self.render_mode = render_mode
        self.failure_prob = failure_prob
        
        self.topology = create_sample_topology(num_nodes)
        self.service_chain = create_sample_service_chain()
        self.current_service_idx = 0
        
        self.action_space = spaces.Discrete(num_nodes)
        
        # State bao gồm (cpu_util, mem_util, is_active) x num_nodes
        obs_dim = (num_nodes * 3) + num_services + num_services
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        self.episode_reward = 0
        self.placement_history = []
    
    def _get_observation(self) -> np.ndarray:
        obs = []
        for node in self.topology.nodes:
            obs.append(node.cpu_used / node.cpu_capacity)
            obs.append(node.memory_used / node.memory_capacity)
            obs.append(1.0 if node.is_active else 0.0)  # MỚI: Thêm trạng thái node
        
        current_service_onehot = np.zeros(self.num_services)
        if self.current_service_idx < self.num_services:
            current_service_onehot[self.current_service_idx] = 1.0
        obs.extend(current_service_onehot)
        
        placement = np.zeros(self.num_services)
        for i, svc in enumerate(self.service_chain.services):
            if svc.placed_on >= 0:
                placement[i] = (svc.placed_on + 1) / self.num_nodes
        obs.extend(placement)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self, service: Microservice, node_id: int, success: bool) -> float:
        if not success:
            return -10.0  
        
        reward = 5.0
        node = self.topology.get_node(node_id)
        
        cpu_util = node.cpu_used / node.cpu_capacity
        mem_util = node.memory_used / node.memory_capacity
        
        if cpu_util > 0.8:
            reward -= (cpu_util - 0.8) * 10
        if mem_util > 0.8:
            reward -= (mem_util - 0.8) * 10
            
        if service.id > 0:
            prev_service = self.service_chain.services[service.id - 1]
            if prev_service.placed_on >= 0:
                latency = self.topology.get_latency(prev_service.placed_on, node_id)
                reward += max(0, 10 - latency)
                
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.current_service_idx >= self.num_services:
            return self._get_observation(), 0.0, True, False, {}
            
        # Logic Node Failure ngẫu nhiên
        node_failed_in_step = False
        if np.random.random() < self.failure_prob:
            active_nodes = [n for n in self.topology.nodes if n.is_active]
            if active_nodes:
                failed_node = np.random.choice(active_nodes)
                failed_node.fail_node()
                node_failed_in_step = True
                
                # Phạt agent nặng nếu node vừa hỏng đang chứa các service quan trọng
                for svc in self.service_chain.services:
                    if svc.placed_on == failed_node.id:
                        self.episode_reward -= 20.0
                        svc.placed_on = -1 # Coi như rớt service
        
        service = self.service_chain.services[self.current_service_idx]
        node = self.topology.get_node(action)
        
        # Thử đặt service lên node
        success = node.allocate(service.cpu_request, service.memory_request)
        
        if success:
            service.placed_on = action
            self.placement_history.append((service.id, action))
        
        reward = self._calculate_reward(service, action, success)
        self.episode_reward += reward
        self.current_service_idx += 1
        terminated = self.current_service_idx >= self.num_services
        
        info = {
            'service_id': service.id,
            'node_id': action,
            'success': success,
            'node_failed_event': node_failed_in_step,
            'episode_reward': self.episode_reward
        }
        
        return self._get_observation(), reward, terminated, False, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.topology.reset()
        for svc in self.service_chain.services:
            svc.placed_on = -1
        
        self.current_service_idx = 0
        self.episode_reward = 0
        self.placement_history = []
        return self._get_observation(), {}
    
    def render(self):
        if self.render_mode in ['human', 'ansi']:
            print("\n" + "=" * 50)
            print("KUBERNETES CLUSTER STATUS")
            print("=" * 50)
            
            for node in self.topology.nodes:
                status = "ACTIVE" if node.is_active else "FAILED"
                if not node.is_active:
                    print(f"\nNode {node.id} [{status}] - Đã mất kết nối!")
                    continue
                    
                cpu_bar = "█" * int(10 * node.cpu_used / node.cpu_capacity)
                cpu_bar += "░" * (10 - len(cpu_bar))
                
                mem_bar = "█" * int(10 * node.memory_used / node.memory_capacity)
                mem_bar += "░" * (10 - len(mem_bar))
                
                services_on_node = [svc.name for svc in self.service_chain.services if svc.placed_on == node.id]
                print(f"\nNode {node.id} [{status}]:")
                print(f"  CPU: [{cpu_bar}] {node.cpu_used:.1f}/{node.cpu_capacity}")
                print(f"  MEM: [{mem_bar}] {node.memory_used:.1f}/{node.memory_capacity}")
                print(f"  Services: {services_on_node}")
            print("\n" + "=" * 50)
    
    def close(self):
        pass

gym.register(
    id='K8sPlacement-v0',
    entry_point='envs.k8s_env:K8sPlacementEnv',
)
