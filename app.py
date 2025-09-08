import os
import json
import uuid
import subprocess
import threading
import time
import re
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
import litellm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Model Configuration from environment variables
API_URL = os.getenv('LLM_API_URL')
API_KEY = os.getenv('LLM_API_KEY')
MODEL = os.getenv('LLM_MODEL')

# Validate required environment variables
if not all([API_URL, API_KEY, MODEL]):
    raise ValueError(
        "Missing required environment variables. "
        "Please copy .env.sample to .env and fill in your LLM configuration."
    )

# Configure LiteLLM
litellm.set_verbose = os.getenv('LITELLM_VERBOSE', 'False').lower() == 'true'
os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_BASE"] = API_URL
# os.environ["OPENAI_API_BASE"] = "https://api.llama.com/compat/v1/"

# Create base directories
os.makedirs("projects", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Global storage
projects = {}
agents = {}
agent_logs = {}

@dataclass
class Task:
    id: str
    description: str
    branch: str
    phase: int = 1
    status: str = "pending"  # pending, in_progress, completed, failed
    assigned_to: Optional[str] = None
    output: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class Agent:
    id: str
    type: str  # "pm" or "swe"
    project_id: str
    status: str = "idle"  # idle, working, completed, failed
    current_task: Optional[str] = None
    thread: Optional[threading.Thread] = None

@dataclass
class Project:
    id: str
    name: str
    description: str
    status: str = "created"  # created, planning, in_progress, completed, failed
    tasks: List[Task] = field(default_factory=list)
    pm_agent_id: Optional[str] = None
    swe_agent_ids: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    project_dir: Optional[str] = None

class BashExecutor:
    """Handles bash command execution with proper error handling and logging"""
    
    @staticmethod
    def run(command: str, cwd: str = None, timeout: int = 30) -> Tuple[bool, str]:
        """Execute a bash command and return (success, output)"""
        print(f"[BASH] Executing in {cwd}: {command}")
        
        try:
            # Ensure directory exists
            if cwd and not os.path.exists(cwd):
                os.makedirs(cwd, exist_ok=True)
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd
            )
            
            output = f"$ {command}\n"
            if result.stdout:
                output += f"{result.stdout}\n"
            if result.stderr:
                output += f"STDERR: {result.stderr}\n"
            output += f"Exit code: {result.returncode}\n"
            
            success = result.returncode == 0
            print(f"[BASH] Result: {'SUCCESS' if success else 'FAILED'}")
            print(f"[BASH] Output: {output[:200]}...")
            
            return success, output
            
        except subprocess.TimeoutExpired:
            output = f"$ {command}\nERROR: Command timed out after {timeout} seconds"
            print(f"[BASH] TIMEOUT: {command}")
            return False, output
        except Exception as e:
            output = f"$ {command}\nERROR: {str(e)}"
            print(f"[BASH] EXCEPTION: {str(e)}")
            return False, output

class SimpleAgent:
    def __init__(self, agent_id: str, agent_type: str, project_id: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.project_id = project_id
        self.agent = agents[agent_id]
        self.project = projects[project_id]
        self.logs = []
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create central log directory
        self.central_log_dir = os.path.join("logs", self.project.name, self.run_id)
        os.makedirs(self.central_log_dir, exist_ok=True)
        
        # Load system prompt
        prompt_file = f"prompts/{agent_type}_agent.txt"
        with open(prompt_file, 'r') as f:
            self.system_prompt = f.read()
        
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.bash = BashExecutor()
    
    def log(self, message: str, level: str = "INFO"):
        """Add a log entry"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        self.logs.append(log_entry)
        agent_logs[self.agent_id] = self.logs
        
        # Print to console
        print(f"[{self.agent_type.upper()}:{self.agent_id[:8]}] {message}")
        
        # Write to log file
        log_filename = f"{self.agent_type}_agent_{self.agent_id[:8]}.log"
        log_path = os.path.join(self.central_log_dir, log_filename)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{log_entry['timestamp']}] {log_entry['level']}: {log_entry['message']}\n")
    
    def extract_bash_commands(self, text: str) -> List[str]:
        """Extract bash commands from LLM response"""
        commands = []
        
        # Find all code blocks with ```bash
        bash_blocks = re.findall(r'```bash\n(.*?)\n```', text, re.DOTALL)
        for block in bash_blocks:
            # Check if this is a grouped command (starts with '(' and contains ')' later)
            block = block.strip()
            if block.startswith('(') and ')' in block:
                # This is a grouped command, keep it together until the closing )
                # Find the index of the last ) followed by > or >>
                redirect_match = re.search(r'\)\s*>+\s*\S+', block)
                if redirect_match:
                    # Extract the full grouped command including the redirect
                    grouped_command = block[:redirect_match.end()]
                    commands.append(grouped_command)
                    
                    # Process any remaining commands after the grouped command
                    remaining = block[redirect_match.end():].strip()
                    if remaining:
                        remaining_lines = remaining.split('\n')
                        for line in remaining_lines:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                if not any(pattern in line.lower() for pattern in [
                                    'switched to', 'already up to date', 'output:', 'error:', 
                                    'volume in drive', 'directory of', 'exit code:'
                                ]):
                                    if not line.startswith('CREATE_SWE_AGENT'):
                                        commands.append(line)
                else:
                    # No redirect found, might be malformed
                    self.log(f"Warning: Grouped command without redirect: {block[:50]}...", "WARNING")
                    # Fall back to line-by-line processing
                    lines = block.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#') and line not in ['(', ')']:
                            commands.append(line)
            else:
                # Regular line-by-line processing
                lines = block.split('\n')
                current_command = []
                
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip comments
                        # Skip lines that look like output (common patterns)
                        if any(pattern in line.lower() for pattern in [
                            'switched to', 'already up to date', 'output:', 'error:', 
                            'volume in drive', 'directory of', 'exit code:'
                        ]):
                            continue
                        
                        # Check if line ends with \ (continuation)
                        if line.endswith('\\'):
                            current_command.append(line[:-1])
                        else:
                            current_command.append(line)
                            # Join multi-line command and add to list
                            if current_command:
                                full_command = ' '.join(current_command)
                                # Don't add CREATE_SWE_AGENT as bash command
                                if not full_command.startswith('CREATE_SWE_AGENT'):
                                    commands.append(full_command)
                                current_command = []
                
                # Add any remaining command
                if current_command:
                    commands.append(' '.join(current_command))
        
        # Debug logging
        if commands:
            self.log(f"Extracted {len(commands)} commands from response")
            for i, cmd in enumerate(commands):
                self.log(f"Command {i+1}: {cmd[:100]}{'...' if len(cmd) > 100 else ''}")
        
        return commands
    
    def extract_file_creations(self, text: str) -> List[Dict]:
        """Extract file creation tags from LLM response"""
        file_creations = []
        
        # Pattern to match <file path="...">content</file>
        file_pattern = re.compile(r'<file\s+path="([^"]+)"\s*>```(?:(\w+)\n)?(.*?)```</file>', re.DOTALL)
        
        for match in file_pattern.finditer(text):
            relative_path = match.group(1)
            language = match.group(2) or ''  # Optional language hint after ```
            content = match.group(3).strip()
            
            file_creations.append({
                'path': relative_path,
                'language': language,
                'content': content
            })
            
            self.log(f"Found file creation: {relative_path} ({len(content)} chars, language: {language or 'auto'})")
        
        return file_creations
    
    def create_files_from_tags(self, file_creations: List[Dict]) -> List[Tuple[bool, str]]:
        """Create files from extracted file tags"""
        results = []
        
        for file_info in file_creations:
            relative_path = file_info['path']
            content = file_info['content']
            
            # Construct full path relative to project directory
            full_path = os.path.join(self.project.project_dir, relative_path)
            
            try:
                # Create directory if it doesn't exist
                dir_path = os.path.dirname(full_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                
                # Write the file
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                output = f"Created file: {relative_path}\nSize: {len(content)} bytes"
                self.log(f"Successfully created file: {relative_path}")
                results.append((True, output))
                
                # Add to git context
                self.messages.append({
                    "role": "user",
                    "content": f"File created: {relative_path}"
                })
                
            except Exception as e:
                output = f"Failed to create file {relative_path}: {str(e)}"
                self.log(output, "ERROR")
                results.append((False, output))
        
        return results
    
    def extract_read_file_calls(self, text: str) -> List[str]:
        """Extract READ_FILE calls from agent response"""
        read_calls = []
        
        # Pattern to match READ_FILE(path="...")
        pattern = re.compile(r'READ_FILE\s*\(\s*path="([^"]+)"\s*\)')
        
        for match in pattern.finditer(text):
            file_path = match.group(1)
            read_calls.append(file_path)
            self.log(f"Found READ_FILE request: {file_path}")
        
        return read_calls
    
    def execute_read_file(self, file_path: str) -> Tuple[bool, str]:
        """Read a file and return its contents"""
        try:
            # Construct full path relative to project directory
            full_path = os.path.join(self.project.project_dir, file_path)
            
            if not os.path.exists(full_path):
                output = f"File not found: {file_path}"
                self.log(output, "WARNING")
                return False, output
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            output = f"Contents of {file_path}:\n{content}"
            self.log(f"Successfully read file: {file_path} ({len(content)} chars)")
            
            # Add full file content to context
            self.messages.append({
                "role": "user",
                "content": output
            })
            
            return True, output
            
        except Exception as e:
            output = f"Failed to read file {file_path}: {str(e)}"
            self.log(output, "ERROR")
            return False, output
    
    def extract_tool_calls(self, text: str) -> List[Dict]:
        """Extract tool calls from PM agent response"""
        tool_calls = []
        
        # Look for CREATE_SWE_AGENT pattern
        swe_matches = re.findall(
            r'CREATE_SWE_AGENT\s*\(\s*task_id="([^"]+)"\s*,\s*description="([^"]+)"\s*,\s*branch="([^"]+)"\s*\)',
            text
        )
        for match in swe_matches:
            tool_calls.append({
                "tool": "create_swe_agent",
                "task_id": match[0],
                "description": match[1],
                "branch": match[2]
            })
        
        return tool_calls
    
    def query_llm(self, message: str, max_retries: int = 3) -> str:
        """Query the LLM and return response with retry logic"""
        self.messages.append({"role": "user", "content": message})
        self.log(f"Querying LLM with: {message[:100]}...")
        
        last_error = None
        for attempt in range(max_retries):
            try:
                # Add exponential backoff for retries
                if attempt > 0:
                    wait_time = 2 ** attempt  # 2, 4, 8 seconds
                    self.log(f"Retry attempt {attempt + 1}/{max_retries} after {wait_time}s delay...", "WARNING")
                    time.sleep(wait_time)
                
                response = litellm.completion(
                    model=MODEL,
                    messages=self.messages,
                    temperature=0.1,
                    api_base=API_URL,
                    api_key=API_KEY,
                    timeout=60  # 60 second timeout
                )
                
                reply = response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": reply})
                self.log(f"LLM Response received: {len(reply)} chars")
                return reply
                
            except Exception as e:
                last_error = e
                error_msg = f"LLM Error (attempt {attempt + 1}/{max_retries}): {str(e)}"
                self.log(error_msg, "ERROR")
                
                # Check if it's a rate limit or temporary error
                if "rate limit" in str(e).lower() or "timeout" in str(e).lower():
                    continue  # Retry with backoff
                elif attempt == max_retries - 1:
                    # Final attempt failed
                    fallback_msg = f"LLM query failed after {max_retries} attempts. Last error: {str(last_error)}"
                    self.log(fallback_msg, "ERROR")
                    # Remove the failed user message to keep context clean
                    if self.messages and self.messages[-1]["role"] == "user":
                        self.messages.pop()
                    return fallback_msg
    
    def execute_bash_commands(self, commands: List[str]) -> List[Tuple[bool, str]]:
        """Execute a list of bash commands"""
        results = []
        project_dir = self.project.project_dir
        
        for cmd in commands:
            self.log(f"Executing: {cmd[:100]}{'...' if len(cmd) > 100 else ''}")
            success, output = self.bash.run(cmd, cwd=project_dir)
            results.append((success, output))
            
            # Add output as context for next commands
            # This ensures the agent knows what happened with each command
            context_message = f"Command executed:\n{output}"
            self.messages.append({
                "role": "user", 
                "content": context_message
            })
            self.log(f"Added command output to context (length: {len(context_message)} chars)")
            
            # Stop on failure
            if not success:
                self.log(f"Command failed: {cmd}", "ERROR")
                break
        
        return results

def run_pm_agent(agent_id: str, project_id: str):
    """Run the PM agent to plan and coordinate the project"""
    agent = SimpleAgent(agent_id, "pm", project_id)
    agents[agent_id].status = "working"
    project = projects[project_id]
    
    try:
        agent.log("Starting PM agent")
        
        # First, create the project directory structure
        project_dir = os.path.abspath(f"projects/{project.name.replace(' ', '_').lower()}")
        project.project_dir = project_dir
        
        # Log directory is now handled by SimpleAgent
        
        # Create directory using Python (more reliable)
        try:
            os.makedirs(project_dir, exist_ok=True)
            agent.log(f"Created project directory: {project_dir}")
        except Exception as e:
            agent.log(f"Failed to create directory: {str(e)}", "ERROR")
            agents[agent_id].status = "failed"
            project.status = "failed"
            return
        
        # Platform-specific git commands
        if os.name == 'nt':  # Windows
            setup_commands = [
                'git init',
                'git config user.name "SWE-Agent"',
                'git config user.email "agent@example.com"',
                f'echo # {project.name} > README.md',
                'git add README.md',
                'git commit -m "Initial commit"'
            ]
        else:  # Linux/Mac
            setup_commands = [
                "git init",
                "git config user.name 'SWE-Agent'",
                "git config user.email 'agent@example.com'",
                f"echo '# {project.name}' > README.md",
                "git add README.md",
                "git commit -m 'Initial commit'"
            ]
        
        agent.log("Setting up git repository")
        for cmd in setup_commands:
            success, output = agent.bash.run(cmd, cwd=project_dir)
            agent.log(f"Setup command result: {output[:100]}")
            if not success:
                agent.log(f"Failed to setup project: {output}", "ERROR")
                agents[agent_id].status = "failed"
                project.status = "failed"
                return
        
        # PHASE 1: Create PRD
        agent.log("PHASE 1: Creating Product Requirements Document")
        prd_prompt = f"""
Project Name: {project.name}
Project Description: {project.description}

Create a COMPREHENSIVE Product Requirements Document (PRD.md) following the template in your instructions.

IMPORTANT: 
- Be extremely detailed and specific about EVERYTHING
- Include complete file structures with exact filenames
- Provide specific implementation details for each component
- List exact CSS classes, color values, and dimensions
- Include code snippets and examples
- Describe user interactions in detail
- Specify exact dependencies and versions needed

After creating the PRD, verify it with:
```bash
type PRD.md
```
"""
        
        response = agent.query_llm(prd_prompt)
        
        # Extract and execute READ_FILE calls
        read_calls = agent.extract_read_file_calls(response)
        for file_path in read_calls:
            agent.execute_read_file(file_path)
        
        # Extract file creations (like PRD.md)
        file_creations = agent.extract_file_creations(response)
        if file_creations:
            agent.create_files_from_tags(file_creations)
        
        # Execute any bash commands
        prd_commands = agent.extract_bash_commands(response)
        if prd_commands:
            agent.execute_bash_commands(prd_commands)
        
        # PHASE 2: Plan implementation phases
        agent.log("PHASE 2: Planning implementation phases")
        planning_prompt = f"""
Based on the PRD you just created, now create SWE agents for implementation.
Remember to organize by phases:
- Phase 1: One agent for initial setup/scaffolding
- Phase 2: Multiple agents for different components
- Phase 3: Integration and finishing touches (if needed)

Use CREATE_SWE_AGENT with proper phase indicators in task_id.
"""
        
        response = agent.query_llm(planning_prompt)
        
        # Extract and execute READ_FILE calls
        read_calls = agent.extract_read_file_calls(response)
        for file_path in read_calls:
            agent.execute_read_file(file_path)
        
        # Extract bash commands and tool calls
        bash_commands = agent.extract_bash_commands(response)
        if bash_commands:
            agent.execute_bash_commands(bash_commands)
        
        # Extract tool calls (SWE agent creation)
        tool_calls = agent.extract_tool_calls(response)
        
        if not tool_calls:
            # Fallback: create default phased tasks
            agent.log("No tool calls found, creating default phased tasks", "WARNING")
            tool_calls = [
                {"tool": "create_swe_agent", "task_id": "phase1_setup", "description": "Create project structure and basic files", "branch": "feature/scaffolding"},
                {"tool": "create_swe_agent", "task_id": "phase2_core", "description": "Implement core functionality", "branch": "feature/core"},
                {"tool": "create_swe_agent", "task_id": "phase2_ui", "description": "Add user interface components", "branch": "feature/ui"}
            ]
        
        # Group tasks by phase
        phases = {}
        for call in tool_calls:
            if call["tool"] == "create_swe_agent":
                # Extract phase from task_id
                phase_num = 1
                if "phase" in call["task_id"]:
                    try:
                        phase_num = int(call["task_id"].split("phase")[1].split("_")[0])
                    except:
                        phase_num = 1
                
                if phase_num not in phases:
                    phases[phase_num] = []
                phases[phase_num].append(call)
        
        # Check for duplicate task_ids and branches
        seen_task_ids = set()
        seen_branches = set()
        for phase_tasks in phases.values():
            for call in phase_tasks:
                if call["task_id"] in seen_task_ids:
                    agent.log(f"WARNING: Duplicate task_id found: {call['task_id']}", "WARNING")
                if call["branch"] in seen_branches:
                    agent.log(f"WARNING: Duplicate branch name found: {call['branch']}", "WARNING")
                seen_task_ids.add(call["task_id"])
                seen_branches.add(call["branch"])
        
        # Execute phases sequentially
        project.status = "in_progress"
        for phase_num in sorted(phases.keys()):
            agent.log(f"Starting Phase {phase_num}")
            phase_tasks = phases[phase_num]
            phase_agents = []
            
            # Create all agents for this phase
            for call in phase_tasks:
                # Create task
                task = Task(
                    id=call["task_id"],
                    description=call["description"],
                    branch=call["branch"],
                    phase=phase_num
                )
                project.tasks.append(task)
                agent.log(f"Created Phase {phase_num} task: {task.description} on branch {task.branch}")
                
                # Create SWE agent
                swe_id = str(uuid.uuid4())
                swe_agent = Agent(
                    id=swe_id,
                    type="swe",
                    project_id=project_id
                )
                agents[swe_id] = swe_agent
                project.swe_agent_ids.append(swe_id)
                task.assigned_to = swe_id
                phase_agents.append((swe_id, task.id))
                
            # Start all agents for this phase
            for swe_id, task_id in phase_agents:
                thread = threading.Thread(
                    target=run_swe_agent,
                    args=(swe_id, project_id, task_id),
                    daemon=True
                )
                thread.start()
                agents[swe_id].thread = thread
                agent.log(f"Started SWE agent {swe_id[:8]} for task {task_id}")
            
            # Wait for this phase to complete
            agent.log(f"Monitoring Phase {phase_num} progress")
            phase_complete = False
            while not phase_complete:
                time.sleep(5)
                
                # Check if all tasks in this phase are completed
                phase_tasks_status = [t for t in project.tasks if t.phase == phase_num]
                phase_complete = all(t.status == "completed" for t in phase_tasks_status)
                
                if not phase_complete:
                    # Check status
                    status_prompt = f"Phase {phase_num} task status:\n"
                    for task in phase_tasks_status:
                        status_prompt += f"- {task.description}: {task.status}\n"
                    
                    status_prompt += "\nProvide bash commands to check progress. You can also use READ_FILE(path=\"filename\") to read specific files:"
                    
                    response = agent.query_llm(status_prompt)
                    
                    # Extract and execute READ_FILE calls
                    read_calls = agent.extract_read_file_calls(response)
                    for file_path in read_calls:
                        agent.execute_read_file(file_path)
                    
                    # Extract and execute bash commands
                    commands = agent.extract_bash_commands(response)
                    if commands[:3]:  # Limit monitoring commands
                        agent.execute_bash_commands(commands[:3])
            
            agent.log(f"Phase {phase_num} completed!")
        
        # All phases completed
        agent.log("All phases completed!")
        agents[agent_id].status = "completed"
        project.status = "completed"
        
    except Exception as e:
        agent.log(f"PM agent failed: {str(e)}", "ERROR")
        agents[agent_id].status = "failed"
        project.status = "failed"

def run_swe_agent(agent_id: str, project_id: str, task_id: str):
    """Run a SWE agent to implement a specific task"""
    agent = SimpleAgent(agent_id, "swe", project_id)
    agents[agent_id].status = "working"
    agents[agent_id].current_task = task_id
    
    project = projects[project_id]
    task = next(t for t in project.tasks if t.id == task_id)
    task.status = "in_progress"
    
    # Log directory is now handled by SimpleAgent
    
    try:
        agent.log(f"Starting SWE agent for task: {task.description}")
        agent.log(f"Task ID: {task.id}, Phase: {task.phase}, Branch: {task.branch}")
        
        # Initial prompt to implement the task
        prompt = f"""
Task: {task.description}
Branch: {task.branch}
Project Directory: {project.project_dir}
Phase: {task.phase}

IMPORTANT: You are currently in the directory: {project.project_dir}
All commands will execute in this directory.

SPECIFIC INSTRUCTIONS:
{task.description}

First, check out your feature branch and implement the task. Use Windows commands to:
1. Create and checkout the branch
2. Implement the required functionality efficiently
3. Test your implementation
4. Commit your changes

Start with these commands:
```bash
git checkout -b {task.branch}
echo %cd%
dir
```
"""
        
        max_steps = 15  # Increased for complex tasks
        for step in range(max_steps):
            agent.log(f"Step {step + 1}/{max_steps}")
            
            # Get LLM response
            if step == 0:
                response = agent.query_llm(prompt)
            else:
                response = agent.query_llm("Continue with the implementation. What's next?")
            
            results = []  # Initialize results for this step
            
            # First, check for READ_FILE calls
            read_calls = agent.extract_read_file_calls(response)
            for file_path in read_calls:
                success, output = agent.execute_read_file(file_path)
                results.append((success, output))
            
            # Then check for file creations
            file_creations = agent.extract_file_creations(response)
            if file_creations:
                file_results = agent.create_files_from_tags(file_creations)
                results.extend(file_results)
            
            # Finally extract and execute bash commands
            commands = agent.extract_bash_commands(response)
            if commands:
                bash_results = agent.execute_bash_commands(commands)
                results.extend(bash_results)
                
                # Check for completion signal
                for success, output in results:
                    if any(signal in output.lower() for signal in ["task completed", "implementation complete", "finished implementation"]):
                        agent.log("Task completed successfully!")
                        
                        # Commit the changes
                        commit_commands = [
                            "git add .",
                            f'git commit -m "Phase {task.phase}: {task.description}"'
                        ]
                        for cmd in commit_commands:
                            agent.bash.run(cmd, cwd=project.project_dir)
                        
                        task.status = "completed"
                        task.output = response
                        agents[agent_id].status = "completed"
                        return
            
            # Check if task seems done based on response
            if any(phrase in response.lower() for phrase in ["task completed", "implementation complete", "done", "finished", "task is complete"]):
                agent.log("Task appears to be completed based on response")
                
                # Final commit
                commit_commands = [
                    "git add .",
                    f'git commit -m "Phase {task.phase}: {task.description}"'
                ]
                for cmd in commit_commands:
                    success, output = agent.bash.run(cmd, cwd=project.project_dir)
                    if success:
                        agent.log(f"Committed: {output[:100]}")
                
                task.status = "completed"
                task.output = response
                agents[agent_id].status = "completed"
                return
        
        # Max steps reached
        agent.log("Max steps reached, marking as completed", "WARNING")
        task.status = "completed"
        agents[agent_id].status = "completed"
        
    except Exception as e:
        agent.log(f"SWE agent failed: {str(e)}", "ERROR")
        task.status = "failed"
        agents[agent_id].status = "failed"

# Flask routes
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/projects', methods=['GET'])
def get_projects():
    return jsonify([asdict(p) for p in projects.values()])

@app.route('/api/projects', methods=['POST'])
def create_project():
    data = request.json
    project_id = str(uuid.uuid4())
    
    # Sanitize project name
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', data['name'])
    
    project = Project(
        id=project_id,
        name=safe_name,
        description=data['description']
    )
    projects[project_id] = project
    
    # Create PM agent
    pm_id = str(uuid.uuid4())
    pm_agent = Agent(
        id=pm_id,
        type="pm",
        project_id=project_id
    )
    agents[pm_id] = pm_agent
    project.pm_agent_id = pm_id
    
    # Start PM agent in background
    thread = threading.Thread(
        target=run_pm_agent,
        args=(pm_id, project_id),
        daemon=True
    )
    thread.start()
    pm_agent.thread = thread
    
    return jsonify(asdict(project))

@app.route('/api/projects/<project_id>', methods=['GET'])
def get_project(project_id):
    if project_id not in projects:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = asdict(projects[project_id])
    
    # Add agent information
    if project_data['pm_agent_id'] and project_data['pm_agent_id'] in agents:
        pm_data = asdict(agents[project_data['pm_agent_id']])
        pm_data['logs'] = agent_logs.get(project_data['pm_agent_id'], [])
        project_data['pm_agent'] = pm_data
    
    project_data['swe_agents'] = []
    for swe_id in project_data['swe_agent_ids']:
        if swe_id in agents:
            swe_data = asdict(agents[swe_id])
            swe_data['logs'] = agent_logs.get(swe_id, [])
            project_data['swe_agents'].append(swe_data)
    
    return jsonify(project_data)

@app.route('/api/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    if project_id not in projects:
        return jsonify({"error": "Project not found"}), 404
    
    project = projects[project_id]
    
    # Stop all agent threads
    if project.pm_agent_id in agents:
        del agents[project.pm_agent_id]
        if project.pm_agent_id in agent_logs:
            del agent_logs[project.pm_agent_id]
    
    for swe_id in project.swe_agent_ids:
        if swe_id in agents:
            del agents[swe_id]
        if swe_id in agent_logs:
            del agent_logs[swe_id]
    
    del projects[project_id]
    return jsonify({"message": "Project deleted"})

@app.route('/api/projects/<project_id>/logs', methods=['GET'])
def get_project_logs(project_id):
    """Get all logs for a project"""
    if project_id not in projects:
        return jsonify({"error": "Project not found"}), 404
    
    project = projects[project_id]
    logs = {}
    
    if project.pm_agent_id in agent_logs:
        logs['pm'] = agent_logs[project.pm_agent_id]
    
    logs['swe'] = {}
    for swe_id in project.swe_agent_ids:
        if swe_id in agent_logs:
            logs['swe'][swe_id] = agent_logs[swe_id]
    
    return jsonify(logs)

if __name__ == '__main__':
    print("Multi-Agent System Starting...")
    print(f"Projects directory: {os.path.abspath('projects')}")
    print(f"Using model: {MODEL}")
    print(f"API endpoint: {API_URL}")
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=True, port=5000)