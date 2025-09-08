# Multi-Agent System

A sophisticated multi-agent system where a Project Manager (PM) agent coordinates multiple Software Engineering (SWE) agents to build software projects. Inspired by mini-swe-agent's philosophy of simplicity and effectiveness.

## Features

- **Project Manager Agent**: Creates detailed PRDs and coordinates implementation phases
- **SWE Agents**: Implement atomic tasks with full context awareness
- **Git-based Collaboration**: Each agent works on separate branches with proper version control
- **Real-time Dashboard**: Web interface to monitor agent progress
- **Comprehensive Logging**: All agent activities logged to `logs/` directory
- **Windows Compatible**: Fully tested on Windows with proper command handling

## Prerequisites

- Python 3.8+
- Git
- pip

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd multiagent
```

2. Install dependencies:
```bash
pip install flask litellm python-dotenv
```

3. Configure your LLM:
```bash
# Copy the sample environment file
cp .env.sample .env

# Edit .env with your LLM configuration
# See .env.sample for examples of different providers
```

## Configuration

The system supports any OpenAI-compatible API. Configure in `.env`:

```env
LLM_API_URL=https://api.example.com/v1/
LLM_API_KEY=your-api-key-here
LLM_MODEL=model-name-here
```

### Supported Providers

- **OpenAI**: GPT-4, GPT-3.5
- **Anthropic**: Claude 3 Opus, Sonnet, Haiku
- **Local Models**: Ollama, LM Studio
- **Cloud Providers**: Cerebras, Together AI, Anyscale
- Any OpenAI-compatible endpoint

## Usage

1. Start the server:
```bash
python app.py
```

2. Open your browser to `http://localhost:5000`

3. Create a new project:
   - Enter project name
   - Provide project description
   - Click "Create Project"

4. Monitor progress:
   - Watch agents create PRD
   - See tasks being assigned
   - Track implementation progress
   - View real-time logs

## How It Works

### Phase 1: Planning
1. PM Agent creates a detailed Product Requirements Document (PRD)
2. PRD includes file structure, technical requirements, and implementation phases
3. PM Agent creates atomic tasks with clear success criteria

### Phase 2: Implementation
1. PM Agent creates SWE agents for each task
2. Each SWE agent:
   - Works on a unique git branch
   - Reads existing files for context
   - Implements only their specific task
   - Commits changes when complete
3. PM Agent monitors progress and verifies completion

### Key Design Principles

- **Atomic Tasks**: Each task does one thing well
- **Full Context**: Agents see all command outputs and maintain conversation history
- **File-based Creation**: Uses `<file>` tags for reliable file creation
- **Verification**: PM can read files to verify implementation

## Project Structure

```
multiagent/
├── app.py                 # Main Flask application
├── prompts/              # Agent system prompts
│   ├── pm_agent.txt      # Project Manager prompt
│   └── swe_agent.txt     # Software Engineer prompt
├── templates/            # Web interface templates
│   └── dashboard.html    # Main dashboard
├── logs/                 # Agent logs (auto-created)
│   └── <project>/        # Per-project logs
│       └── <timestamp>/  # Per-run logs
└── projects/             # Created projects (auto-created)
```

## Logging

All agent activities are logged to `logs/<project_name>/<timestamp>/`:
- `pm_agent_*.log`: Project Manager activities
- `swe_agent_*.log`: Individual SWE agent activities

## Troubleshooting

### "Missing required environment variables"
- Ensure you've copied `.env.sample` to `.env`
- Fill in all required LLM configuration values

### Empty directories not tracked by git
- The system automatically creates `.gitkeep` files in empty directories
- This ensures git can track the directory structure

### Windows command issues
- The system uses Windows-compatible commands
- File creation uses `<file>` tags, not echo commands

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Your chosen license]

## Acknowledgments

Inspired by [mini-swe-agent](https://github.com/o1-labs/mini-swe-agent) and its philosophy of simplicity and effectiveness.