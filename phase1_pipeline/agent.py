from typing import TypedDict
from langgraph.graph import StateGraph, END
from phase1_pipeline.youtube_tools import (
    search_channel_playlists,
    get_videos_from_playlist,
    fetch_all_transcripts
)
from phase1_pipeline.vector_store import store_all_transcripts, get_stored_video_ids


class PipelineState(TypedDict):
    playlists: list[dict]
    videos: list[dict]
    transcripts: list[dict]
    status: str


def node_search_playlists(state: PipelineState) -> PipelineState:
    print("\n[NODE] Searching Campus X playlists...")
    playlists = search_channel_playlists(channel_query="Campus X", max_results=20)
    return {**state, "playlists": playlists, "status": "playlists_fetched"}


def node_get_videos(state: PipelineState) -> PipelineState:
    print("\n[NODE] Extracting videos from playlists...")
    all_videos = []
    for playlist in state["playlists"]:
        print(f"  → Playlist: {playlist['title']}")
        videos = get_videos_from_playlist(playlist["playlist_id"])
        all_videos.extend(videos)
    print(f"Total videos found: {len(all_videos)}")
    return {**state, "videos": all_videos, "status": "videos_fetched"}


def node_fetch_transcripts(state: PipelineState) -> PipelineState:
    print("\n[NODE] Fetching transcripts...")
    stored = get_stored_video_ids()
    transcripts = fetch_all_transcripts(state["videos"], skip_ids=stored)
    print(f"Transcripts fetched: {len(transcripts)}")
    return {**state, "transcripts": transcripts, "status": "transcripts_fetched"}


def node_store_vectors(state: PipelineState) -> PipelineState:
    print("\n[NODE] Storing into ChromaDB...")
    store_all_transcripts(state["transcripts"])
    return {**state, "status": "done"}


def build_pipeline_agent():
    graph = StateGraph(PipelineState)

    graph.add_node("search_playlists", node_search_playlists)
    graph.add_node("get_videos", node_get_videos)
    graph.add_node("fetch_transcripts", node_fetch_transcripts)
    graph.add_node("store_vectors", node_store_vectors)

    graph.set_entry_point("search_playlists")
    graph.add_edge("search_playlists", "get_videos")
    graph.add_edge("get_videos", "fetch_transcripts")
    graph.add_edge("fetch_transcripts", "store_vectors")
    graph.add_edge("store_vectors", END)

    return graph.compile()
