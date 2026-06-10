import os
# Must run before ANY import that pulls in huggingface_hub, or the offline
# flag is ignored and the run hangs on a network model-file check.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
sys.stdout.reconfigure(encoding="utf-8")

from phase1_pipeline.agent import build_pipeline_agent

if __name__ == "__main__":
    print("Starting Campus X Data Pipeline...\n")

    agent = build_pipeline_agent()

    initial_state = {
        "playlists": [],
        "videos": [],
        "transcripts": [],
        "status": "start"
    }

    final_state = agent.invoke(initial_state)
    print(f"\nPipeline complete! Status: {final_state['status']}")
    print(f"Total transcripts stored: {len(final_state['transcripts'])}")
