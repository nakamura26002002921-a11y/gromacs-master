# main.py
import asyncio
from gromacs_agent.core.graph import agent_app
from gromacs_agent.utils.logger import setup_logger

async def main():
    setup_logger()
    
    print("🚀 Starting GROMACS Autonomous Agent...")
    
    initial_state = {
        "system_name": "lysozyme_in_water",
        "pdb_file": "1AKI.pdb",
        "workflow": [], # Plannerが生成
        "step_index": 0,
        "attempt_count": 0,
        "max_attempts": 3,
        "current_config": {
            "force_field": "amber99sb-ildn",
            "water": "tip3p",
            "dt": 0.002,
            "nsteps": 50000
        },
        "history": []
    }
    
    # グラフの実行
    result = agent_app.invoke(initial_state)
    
    print(f"✅ Final Status: {result['status']}")
    print(f"🔧 Final Config: {result['current_config']}")

if __name__ == "__main__":
    asyncio.run(main())
