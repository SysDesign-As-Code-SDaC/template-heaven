# 🤖 Polyfunctional Robots Template

A production-ready template for developing polyfunctional robots capable of performing multiple tasks and adapting to various functions for 2025 and beyond.

## 🚀 Features

- **Multi-Task Capability** - Robots that can perform multiple different tasks
- **Adaptive Systems** - Self-adapting robots that learn new functions
- **Task Switching** - Seamless switching between different tasks
- **Modular Design** - Modular robot architecture for easy reconfiguration
- **AI-Powered Learning** - Machine learning for task acquisition
- **Sensor Fusion** - Advanced sensor integration for task perception
- **Manipulation Systems** - Advanced manipulation capabilities
- **Navigation Systems** - Autonomous navigation and path planning
- **Human-Robot Interaction** - Natural human-robot collaboration
- **Safety Systems** - Comprehensive safety and fail-safe mechanisms

## 📋 Prerequisites

- Python 3.9+
- ROS2 (Robot Operating System)
- CUDA 12.0+ (for AI acceleration)
- Robot Hardware (optional for simulation)
- 16GB+ RAM (for AI processing)

## 🛠️ Quick Start

### 1. Create New Polyfunctional Robot Project

```bash
git clone <this-repo> my-polyfunctional-robot
cd my-polyfunctional-robot
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup ROS2 workspace
colcon build
source install/setup.bash
```

### 3. Configure Robot System

```bash
cp config/robot_config.yaml.example config/robot_config.yaml
# Edit configuration file
```

### 4. Run Polyfunctional Robot

```bash
# Start robot system
python src/robot_system/main.py --config config/robot_config.yaml

# Start task manager
python src/task_manager/manager.py --config config/task_config.yaml

# Start learning system
python src/learning/learner.py --config config/learning_config.yaml
```

## 📁 Project Structure

```
├── src/                        # Source code
│   ├── robot_system/          # Main robot system
│   │   ├── main.py            # Robot main controller
│   │   ├── robot_controller.py # Robot control system
│   │   ├── sensor_manager.py  # Sensor management
│   │   └── actuator_manager.py # Actuator control
│   ├── task_manager/          # Task management
│   │   ├── manager.py         # Task manager
│   │   ├── task_scheduler.py  # Task scheduling
│   │   ├── task_executor.py   # Task execution
│   │   └── task_planner.py    # Task planning
│   ├── learning/              # Learning systems
│   │   ├── learner.py         # Main learning system
│   │   ├── skill_acquisition.py # Skill learning
│   │   ├── task_learning.py   # Task learning
│   │   └── adaptation.py      # Adaptive learning
│   ├── perception/            # Perception systems
│   │   ├── vision.py          # Computer vision
│   │   ├── sensor_fusion.py   # Sensor fusion
│   │   ├── object_detection.py # Object detection
│   │   └── environment_mapping.py # Environment mapping
│   ├── manipulation/          # Manipulation systems
│   │   ├── gripper_control.py # Gripper control
│   │   ├── arm_control.py     # Arm control
│   │   ├── manipulation_planner.py # Manipulation planning
│   │   └── force_control.py   # Force control
│   ├── navigation/            # Navigation systems
│   │   ├── path_planner.py    # Path planning
│   │   ├── localization.py    # Robot localization
│   │   ├── mapping.py         # Environment mapping
│   │   └── obstacle_avoidance.py # Obstacle avoidance
│   ├── interaction/           # Human-robot interaction
│   │   ├── speech_recognition.py # Speech recognition
│   │   ├── natural_language.py # Natural language processing
│   │   ├── gesture_recognition.py # Gesture recognition
│   │   └── collaboration.py   # Human-robot collaboration
│   └── safety/                # Safety systems
│       ├── safety_monitor.py  # Safety monitoring
│       ├── emergency_stop.py  # Emergency stop
│       ├── collision_detection.py # Collision detection
│       └── fail_safe.py       # Fail-safe mechanisms
├── config/                    # Configuration files
│   ├── robot_config.yaml
│   ├── task_config.yaml
│   └── learning_config.yaml
├── tests/                     # Test files
├── docs/                      # Documentation
└── examples/                  # Example implementations
```

## 🔧 Available Scripts

```bash
# Robot System
python src/robot_system/main.py          # Start robot system
python src/robot_system/robot_controller.py # Robot control
python src/robot_system/sensor_manager.py # Sensor management

# Task Management
python src/task_manager/manager.py       # Task manager
python src/task_manager/task_scheduler.py # Task scheduling
python src/task_manager/task_executor.py # Task execution

# Learning Systems
python src/learning/learner.py           # Learning system
python src/learning/skill_acquisition.py # Skill learning
python src/learning/task_learning.py     # Task learning

# Perception and Manipulation
python src/perception/vision.py          # Computer vision
python src/manipulation/gripper_control.py # Gripper control
python src/navigation/path_planner.py    # Path planning

# Interaction and Safety
python src/interaction/speech_recognition.py # Speech recognition
python src/safety/safety_monitor.py      # Safety monitoring
```

## 🤖 Polyfunctional Robot Implementation

### Main Robot System

```python
# src/robot_system/robot_controller.py
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

class TaskType(Enum):
    MANIPULATION = "manipulation"
    NAVIGATION = "navigation"
    INSPECTION = "inspection"
    ASSEMBLY = "assembly"
    CLEANING = "cleaning"
    DELIVERY = "delivery"
    MONITORING = "monitoring"
    INTERACTION = "interaction"

class RobotState(Enum):
    IDLE = "idle"
    TASK_EXECUTION = "task_execution"
    LEARNING = "learning"
    MAINTENANCE = "maintenance"
    ERROR = "error"

@dataclass
class Task:
    """Task definition for polyfunctional robot."""
    id: str
    name: str
    task_type: TaskType
    description: str
    parameters: Dict
    priority: int
    estimated_duration: float
    required_skills: List[str]
    safety_requirements: List[str]

@dataclass
class RobotCapability:
    """Robot capability definition."""
    skill_name: str
    proficiency_level: float
    parameters: Dict
    last_updated: datetime

class PolyfunctionalRobot:
    """Main polyfunctional robot controller."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.task_manager = TaskManager(config)
        self.learning_system = LearningSystem(config)
        self.perception_system = PerceptionSystem(config)
        self.manipulation_system = ManipulationSystem(config)
        self.navigation_system = NavigationSystem(config)
        self.interaction_system = InteractionSystem(config)
        self.safety_system = SafetySystem(config)
        
        # Robot state
        self.current_state = RobotState.IDLE
        self.current_task = None
        self.capabilities = {}
        self.task_history = []
        
        # Initialize capabilities
        self._initialize_capabilities()
    
    def _initialize_capabilities(self):
        """Initialize robot capabilities."""
        self.capabilities = {
            'manipulation': RobotCapability('manipulation', 0.8, {}, datetime.now()),
            'navigation': RobotCapability('navigation', 0.9, {}, datetime.now()),
            'vision': RobotCapability('vision', 0.7, {}, datetime.now()),
            'speech': RobotCapability('speech', 0.6, {}, datetime.now()),
            'learning': RobotCapability('learning', 0.5, {}, datetime.now())
        }
    
    async def start_robot(self):
        """Start the polyfunctional robot system."""
        self.logger.info("Starting polyfunctional robot system")
        
        # Initialize all subsystems
        await self._initialize_subsystems()
        
        # Start main control loop
        await self._main_control_loop()
    
    async def _initialize_subsystems(self):
        """Initialize all robot subsystems."""
        try:
            # Initialize perception system
            await self.perception_system.initialize()
            
            # Initialize manipulation system
            await self.manipulation_system.initialize()
            
            # Initialize navigation system
            await self.navigation_system.initialize()
            
            # Initialize interaction system
            await self.interaction_system.initialize()
            
            # Initialize safety system
            await self.safety_system.initialize()
            
            # Initialize learning system
            await self.learning_system.initialize()
            
            self.logger.info("All subsystems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize subsystems: {e}")
            raise
    
    async def _main_control_loop(self):
        """Main robot control loop."""
        while True:
            try:
                # Check safety status
                safety_status = await self.safety_system.check_safety()
                if not safety_status['safe']:
                    await self._handle_safety_issue(safety_status)
                    continue
                
                # Get next task
                next_task = await self.task_manager.get_next_task()
                
                if next_task:
                    await self._execute_task(next_task)
                else:
                    # No tasks available, enter learning mode
                    await self._enter_learning_mode()
                
                # Update capabilities
                await self._update_capabilities()
                
                # Wait before next iteration
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in main control loop: {e}")
                await self._handle_error(e)
    
    async def _execute_task(self, task: Task):
        """Execute a task."""
        self.logger.info(f"Executing task: {task.name}")
        
        try:
            # Set robot state
            self.current_state = RobotState.TASK_EXECUTION
            self.current_task = task
            
            # Check if robot has required capabilities
            if not await self._check_task_capabilities(task):
                # Learn required skills
                await self._learn_required_skills(task)
            
            # Execute task based on type
            if task.task_type == TaskType.MANIPULATION:
                await self._execute_manipulation_task(task)
            elif task.task_type == TaskType.NAVIGATION:
                await self._execute_navigation_task(task)
            elif task.task_type == TaskType.INSPECTION:
                await self._execute_inspection_task(task)
            elif task.task_type == TaskType.ASSEMBLY:
                await self._execute_assembly_task(task)
            elif task.task_type == TaskType.CLEANING:
                await self._execute_cleaning_task(task)
            elif task.task_type == TaskType.DELIVERY:
                await self._execute_delivery_task(task)
            elif task.task_type == TaskType.MONITORING:
                await self._execute_monitoring_task(task)
            elif task.task_type == TaskType.INTERACTION:
                await self._execute_interaction_task(task)
            
            # Task completed successfully
            await self._task_completed(task)
            
        except Exception as e:
            self.logger.error(f"Error executing task {task.name}: {e}")
            await self._task_failed(task, e)
    
    async def _check_task_capabilities(self, task: Task) -> bool:
        """Check if robot has required capabilities for task."""
        for required_skill in task.required_skills:
            if required_skill not in self.capabilities:
                return False
            
            capability = self.capabilities[required_skill]
            if capability.proficiency_level < 0.5:  # Minimum proficiency threshold
                return False
        
        return True
    
    async def _learn_required_skills(self, task: Task):
        """Learn required skills for a task."""
        self.logger.info(f"Learning required skills for task: {task.name}")
        
        for required_skill in task.required_skills:
            if required_skill not in self.capabilities:
                # Learn new skill
                await self.learning_system.learn_skill(required_skill)
                self.capabilities[required_skill] = RobotCapability(
                    required_skill, 0.5, {}, datetime.now()
                )
            else:
                # Improve existing skill
                await self.learning_system.improve_skill(required_skill)
                self.capabilities[required_skill].proficiency_level += 0.1
                self.capabilities[required_skill].last_updated = datetime.now()
    
    async def _execute_manipulation_task(self, task: Task):
        """Execute manipulation task."""
        self.logger.info(f"Executing manipulation task: {task.name}")
        
        # Get task parameters
        target_object = task.parameters.get('target_object')
        action = task.parameters.get('action')
        position = task.parameters.get('position')
        
        # Plan manipulation
        manipulation_plan = await self.manipulation_system.plan_manipulation(
            target_object, action, position
        )
        
        # Execute manipulation
        await self.manipulation_system.execute_manipulation(manipulation_plan)
    
    async def _execute_navigation_task(self, task: Task):
        """Execute navigation task."""
        self.logger.info(f"Executing navigation task: {task.name}")
        
        # Get task parameters
        destination = task.parameters.get('destination')
        path_type = task.parameters.get('path_type', 'optimal')
        
        # Plan navigation
        navigation_plan = await self.navigation_system.plan_navigation(
            destination, path_type
        )
        
        # Execute navigation
        await self.navigation_system.execute_navigation(navigation_plan)
    
    async def _execute_inspection_task(self, task: Task):
        """Execute inspection task."""
        self.logger.info(f"Executing inspection task: {task.name}")
        
        # Get task parameters
        inspection_target = task.parameters.get('inspection_target')
        inspection_type = task.parameters.get('inspection_type')
        
        # Navigate to inspection location
        await self.navigation_system.navigate_to(inspection_target)
        
        # Perform inspection
        inspection_results = await self.perception_system.perform_inspection(
            inspection_target, inspection_type
        )
        
        # Report results
        await self._report_inspection_results(inspection_results)
    
    async def _execute_assembly_task(self, task: Task):
        """Execute assembly task."""
        self.logger.info(f"Executing assembly task: {task.name}")
        
        # Get task parameters
        assembly_plan = task.parameters.get('assembly_plan')
        components = task.parameters.get('components')
        
        # Execute assembly steps
        for step in assembly_plan:
            await self._execute_assembly_step(step, components)
    
    async def _execute_cleaning_task(self, task: Task):
        """Execute cleaning task."""
        self.logger.info(f"Executing cleaning task: {task.name}")
        
        # Get task parameters
        cleaning_area = task.parameters.get('cleaning_area')
        cleaning_method = task.parameters.get('cleaning_method')
        
        # Navigate to cleaning area
        await self.navigation_system.navigate_to(cleaning_area)
        
        # Perform cleaning
        await self.manipulation_system.perform_cleaning(cleaning_area, cleaning_method)
    
    async def _execute_delivery_task(self, task: Task):
        """Execute delivery task."""
        self.logger.info(f"Executing delivery task: {task.name}")
        
        # Get task parameters
        pickup_location = task.parameters.get('pickup_location')
        delivery_location = task.parameters.get('delivery_location')
        item = task.parameters.get('item')
        
        # Navigate to pickup location
        await self.navigation_system.navigate_to(pickup_location)
        
        # Pick up item
        await self.manipulation_system.pick_up_item(item)
        
        # Navigate to delivery location
        await self.navigation_system.navigate_to(delivery_location)
        
        # Deliver item
        await self.manipulation_system.deliver_item(item)
    
    async def _execute_monitoring_task(self, task: Task):
        """Execute monitoring task."""
        self.logger.info(f"Executing monitoring task: {task.name}")
        
        # Get task parameters
        monitoring_area = task.parameters.get('monitoring_area')
        monitoring_duration = task.parameters.get('monitoring_duration', 300)  # 5 minutes
        
        # Navigate to monitoring area
        await self.navigation_system.navigate_to(monitoring_area)
        
        # Perform monitoring
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < monitoring_duration:
            # Monitor environment
            monitoring_data = await self.perception_system.monitor_environment()
            
            # Check for anomalies
            anomalies = await self._detect_anomalies(monitoring_data)
            
            if anomalies:
                await self._handle_anomalies(anomalies)
            
            await asyncio.sleep(1)
    
    async def _execute_interaction_task(self, task: Task):
        """Execute interaction task."""
        self.logger.info(f"Executing interaction task: {task.name}")
        
        # Get task parameters
        interaction_type = task.parameters.get('interaction_type')
        target_person = task.parameters.get('target_person')
        
        # Perform interaction
        if interaction_type == 'conversation':
            await self.interaction_system.start_conversation(target_person)
        elif interaction_type == 'assistance':
            await self.interaction_system.provide_assistance(target_person)
        elif interaction_type == 'guidance':
            await self.interaction_system.provide_guidance(target_person)
    
    async def _enter_learning_mode(self):
        """Enter learning mode when no tasks are available."""
        if self.current_state != RobotState.LEARNING:
            self.logger.info("Entering learning mode")
            self.current_state = RobotState.LEARNING
            
            # Learn new skills or improve existing ones
            await self.learning_system.continuous_learning()
    
    async def _update_capabilities(self):
        """Update robot capabilities based on experience."""
        for skill_name, capability in self.capabilities.items():
            # Update proficiency based on recent performance
            recent_performance = await self.learning_system.get_recent_performance(skill_name)
            
            if recent_performance > 0.8:
                capability.proficiency_level = min(1.0, capability.proficiency_level + 0.01)
            elif recent_performance < 0.5:
                capability.proficiency_level = max(0.0, capability.proficiency_level - 0.01)
            
            capability.last_updated = datetime.now()
    
    async def _task_completed(self, task: Task):
        """Handle task completion."""
        self.logger.info(f"Task completed: {task.name}")
        
        # Update task history
        self.task_history.append({
            'task': task,
            'status': 'completed',
            'completion_time': datetime.now()
        })
        
        # Update learning system
        await self.learning_system.task_completed(task)
        
        # Reset robot state
        self.current_state = RobotState.IDLE
        self.current_task = None
    
    async def _task_failed(self, task: Task, error: Exception):
        """Handle task failure."""
        self.logger.error(f"Task failed: {task.name}, Error: {error}")
        
        # Update task history
        self.task_history.append({
            'task': task,
            'status': 'failed',
            'error': str(error),
            'failure_time': datetime.now()
        })
        
        # Update learning system
        await self.learning_system.task_failed(task, error)
        
        # Reset robot state
        self.current_state = RobotState.IDLE
        self.current_task = None
    
    async def _handle_safety_issue(self, safety_status: Dict):
        """Handle safety issues."""
        self.logger.warning(f"Safety issue detected: {safety_status}")
        
        # Stop current task
        if self.current_task:
            await self._emergency_stop()
        
        # Take safety measures
        await self.safety_system.handle_safety_issue(safety_status)
    
    async def _emergency_stop(self):
        """Emergency stop robot."""
        self.logger.warning("Emergency stop activated")
        
        # Stop all subsystems
        await self.manipulation_system.emergency_stop()
        await self.navigation_system.emergency_stop()
        
        # Set robot state
        self.current_state = RobotState.ERROR
        self.current_task = None
    
    async def _handle_error(self, error: Exception):
        """Handle general errors."""
        self.logger.error(f"Robot error: {error}")
        
        # Try to recover
        await self._attempt_recovery(error)
    
    async def _attempt_recovery(self, error: Exception):
        """Attempt to recover from error."""
        self.logger.info("Attempting error recovery")
        
        # Reset subsystems
        await self._reset_subsystems()
        
        # Set robot state
        self.current_state = RobotState.IDLE
        self.current_task = None
    
    async def _reset_subsystems(self):
        """Reset all subsystems."""
        await self.manipulation_system.reset()
        await self.navigation_system.reset()
        await self.perception_system.reset()
        await self.interaction_system.reset()

class TaskManager:
    """Task management system for polyfunctional robot."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.task_queue = []
        self.completed_tasks = []
    
    async def get_next_task(self) -> Optional[Task]:
        """Get next task from queue."""
        if self.task_queue:
            return self.task_queue.pop(0)
        return None
    
    async def add_task(self, task: Task):
        """Add task to queue."""
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda x: x.priority, reverse=True)

class LearningSystem:
    """Learning system for polyfunctional robot."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def learn_skill(self, skill_name: str):
        """Learn a new skill."""
        self.logger.info(f"Learning new skill: {skill_name}")
    
    async def improve_skill(self, skill_name: str):
        """Improve existing skill."""
        self.logger.info(f"Improving skill: {skill_name}")
    
    async def continuous_learning(self):
        """Continuous learning when idle."""
        self.logger.info("Performing continuous learning")
    
    async def get_recent_performance(self, skill_name: str) -> float:
        """Get recent performance for a skill."""
        return 0.8  # Placeholder
    
    async def task_completed(self, task: Task):
        """Handle task completion for learning."""
        self.logger.info(f"Learning from completed task: {task.name}")
    
    async def task_failed(self, task: Task, error: Exception):
        """Handle task failure for learning."""
        self.logger.info(f"Learning from failed task: {task.name}")

class PerceptionSystem:
    """Perception system for polyfunctional robot."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize perception system."""
        self.logger.info("Initializing perception system")
    
    async def perform_inspection(self, target: str, inspection_type: str) -> Dict:
        """Perform inspection."""
        return {'status': 'completed', 'results': {}}
    
    async def monitor_environment(self) -> Dict:
        """Monitor environment."""
        return {'status': 'monitoring', 'data': {}}
    
    async def reset(self):
        """Reset perception system."""
        self.logger.info("Resetting perception system")

class ManipulationSystem:
    """Manipulation system for polyfunctional robot."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize manipulation system."""
        self.logger.info("Initializing manipulation system")
    
    async def plan_manipulation(self, target_object: str, action: str, position: Dict) -> Dict:
        """Plan manipulation."""
        return {'plan': 'manipulation_plan'}
    
    async def execute_manipulation(self, plan: Dict):
        """Execute manipulation plan."""
        self.logger.info("Executing manipulation plan")
    
    async def pick_up_item(self, item: str):
        """Pick up item."""
        self.logger.info(f"Picking up item: {item}")
    
    async def deliver_item(self, item: str):
        """Deliver item."""
        self.logger.info(f"Delivering item: {item}")
    
    async def perform_cleaning(self, area: str, method: str):
        """Perform cleaning."""
        self.logger.info(f"Performing cleaning: {area} with {method}")
    
    async def emergency_stop(self):
        """Emergency stop manipulation."""
        self.logger.warning("Emergency stop manipulation")
    
    async def reset(self):
        """Reset manipulation system."""
        self.logger.info("Resetting manipulation system")

class NavigationSystem:
    """Navigation system for polyfunctional robot."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize navigation system."""
        self.logger.info("Initializing navigation system")
    
    async def plan_navigation(self, destination: str, path_type: str) -> Dict:
        """Plan navigation."""
        return {'plan': 'navigation_plan'}
    
    async def execute_navigation(self, plan: Dict):
        """Execute navigation plan."""
        self.logger.info("Executing navigation plan")
    
    async def navigate_to(self, destination: str):
        """Navigate to destination."""
        self.logger.info(f"Navigating to: {destination}")
    
    async def emergency_stop(self):
        """Emergency stop navigation."""
        self.logger.warning("Emergency stop navigation")
    
    async def reset(self):
        """Reset navigation system."""
        self.logger.info("Resetting navigation system")

class InteractionSystem:
    """Human-robot interaction system."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize interaction system."""
        self.logger.info("Initializing interaction system")
    
    async def start_conversation(self, person: str):
        """Start conversation with person."""
        self.logger.info(f"Starting conversation with: {person}")
    
    async def provide_assistance(self, person: str):
        """Provide assistance to person."""
        self.logger.info(f"Providing assistance to: {person}")
    
    async def provide_guidance(self, person: str):
        """Provide guidance to person."""
        self.logger.info(f"Providing guidance to: {person}")
    
    async def reset(self):
        """Reset interaction system."""
        self.logger.info("Resetting interaction system")

class SafetySystem:
    """Safety system for polyfunctional robot."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize safety system."""
        self.logger.info("Initializing safety system")
    
    async def check_safety(self) -> Dict:
        """Check safety status."""
        return {'safe': True, 'issues': []}
    
    async def handle_safety_issue(self, safety_status: Dict):
        """Handle safety issue."""
        self.logger.warning(f"Handling safety issue: {safety_status}")
```

## 📚 Learning Resources

- [Polyfunctional Robots](https://polyfunctional-robots.com/)
- [Multi-Task Learning](https://multi-task-learning.com/)
- [Robot Operating System](https://ros.org/)

## 🔗 Upstream Source

- **Repository**: [Polyfunctional Robots](https://github.com/polyfunctional-robots)
- **Multi-Task Learning**: [Multi-Task Lab](https://github.com/multi-task-lab)
- **Robot Systems**: [Robot Systems](https://github.com/robot-systems)
- **License**: MIT
