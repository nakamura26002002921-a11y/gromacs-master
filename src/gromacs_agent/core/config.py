# src/gromacs_agent/core/config.py
from pydantic import BaseModel, Field
from typing import Dict, Any

class GromacsConfig(BaseModel):
    force_field: str = Field(default="amber99sb-ildn")
    water_model: str = Field(default="tip3p")
    
class SimulationStepConfig(BaseModel):
    nsteps: int
    dt: float = 0.002
    emtol: float = 1000.0
    
class AgentConfig(BaseModel):
    max_attempts: int = Field(default=3, description="最大リトライ回数")
    llm_provider: str = Field(default="openai", description="openai or anthropic")
    model_name: str = Field(default="gpt-4o")
