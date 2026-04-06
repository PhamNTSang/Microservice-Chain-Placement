import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Node:
    """Đại diện cho một node trong Kubernetes cluster."""
    id: int
    cpu_capacity: float      
    memory_capacity: float   
    bandwidth: float         
    
    cpu_used: float = 0.0
    memory_used: float = 0.0
    is_active: bool = True  # trạng thái sống/chết của Node
    
    @property
    def cpu_available(self) -> float:
        return self.cpu_capacity - self.cpu_used
    
    @property
    def memory_available(self) -> float:
        return self.memory_capacity - self.memory_used
    
    def can_allocate(self, cpu: float, memory: float) -> bool:
        """Kiểm tra node có đủ tài nguyên và đang hoạt động không."""
        if not self.is_active:
            return False
        return self.cpu_available >= cpu and self.memory_available >= memory
    
    def allocate(self, cpu: float, memory: float) -> bool:
        """Cấp phát tài nguyên cho microservice."""
        if not self.can_allocate(cpu, memory):
            return False
        self.cpu_used += cpu
        self.memory_used += memory
        return True
    
    def deallocate(self, cpu: float, memory: float):
        """Giải phóng tài nguyên."""
        self.cpu_used = max(0, self.cpu_used - cpu)
        self.memory_used = max(0, self.memory_used - memory)

    def fail_node(self):
        """Mô phỏng node bị hỏng, giải phóng toàn bộ tài nguyên."""
        self.is_active = False
        self.cpu_used = 0.0
        self.memory_used = 0.0
    
    def reset(self):
        """Reset node về trạng thái ban đầu."""
        self.cpu_used = 0.0
        self.memory_used = 0.0
        self.is_active = True

@dataclass
class Microservice:
    """Đại diện cho một microservice."""
    id: int
    name: str
    cpu_request: float       
    memory_request: float    
    placed_on: int = -1

class ServiceChain:
    """Đại diện cho một chuỗi microservice."""
    def __init__(self, chain_id: int, services: List[Microservice]):
        self.chain_id = chain_id
        self.services = services
        self.latency_requirements = {}
    
    def add_latency_requirement(self, src: int, dst: int, max_latency: float):
        self.latency_requirements[(src, dst)] = max_latency

class NetworkTopology:
    """Mô hình hóa topology mạng của cluster."""
    def __init__(self, num_nodes: int, config: Dict = None):
        self.num_nodes = num_nodes
        self.nodes: List[Node] = []
        self.latency_matrix = np.zeros((num_nodes, num_nodes))
        
        self._initialize_nodes(config)
        self._initialize_latency_matrix()
    
    def _initialize_nodes(self, config: Dict = None):
        default_config = {'cpu_capacity': 8, 'memory_capacity': 16, 'bandwidth': 1000}
        config = config or default_config
        
        for i in range(self.num_nodes):
            node = Node(
                id=i,
                cpu_capacity=config.get('cpu_capacity', 8),
                memory_capacity=config.get('memory_capacity', 16),
                bandwidth=config.get('bandwidth', 1000)
            )
            self.nodes.append(node)
    
    def _initialize_latency_matrix(self):
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    self.latency_matrix[i][j] = 0
                else:
                    self.latency_matrix[i][j] = np.random.uniform(1, 10)
    
    def get_latency(self, src_node: int, dst_node: int) -> float:
        return self.latency_matrix[src_node][dst_node]
    
    def get_node(self, node_id: int) -> Node:
        return self.nodes[node_id]
    
    def reset(self):
        for node in self.nodes:
            node.reset()

def create_sample_topology(num_nodes: int = 5) -> NetworkTopology:
    config = {'cpu_capacity': 8, 'memory_capacity': 16, 'bandwidth': 1000}
    return NetworkTopology(num_nodes, config)

def create_sample_service_chain() -> ServiceChain:
    services = [
        Microservice(0, "api-gateway", 0.5, 1.0),
        Microservice(1, "auth-service", 0.3, 0.5),
        Microservice(2, "user-service", 0.5, 1.0),
        Microservice(3, "order-service", 0.8, 1.5),
        Microservice(4, "database", 1.0, 2.0),
    ]
    chain = ServiceChain(0, services)
    chain.add_latency_requirement(0, 1, 5.0)
    chain.add_latency_requirement(1, 2, 5.0)
    chain.add_latency_requirement(2, 3, 10.0)
    chain.add_latency_requirement(3, 4, 10.0)
    return chain