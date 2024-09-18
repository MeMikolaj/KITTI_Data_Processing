from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Type

import pandas as pd
from tqdm import tqdm

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures.agent import (
    Agent,
    AgentMetadata,
    AgentType,
    FixedExtent,
    VariableExtent,
)

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures.agent import AgentMetadata, AgentType, FixedExtent
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.scene_metadata import Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import EUPedsRecord
from trajdata.utils import arr_utils

TRAIN_SCENES: Final[List[str]] = [
    "0020",
    "0002",
    "0003",
    "0004",
    "0005",
    "0006",
    "0018"
]

VAL_SCENES: Final[List[str]] = [
    "0001"
]

TEST_SCENES: Final[List[str]] = [
    "0000"
]

KITTI_DT: Final[float] = 0.05

class KittiDataset(RawDataset):

    ################################## DONE - NOT TESTED ##################################
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        scene_splits: Dict[str, List[str]] = {
            "train": [name for name in TRAIN_SCENES],
            "val": [name for name in VAL_SCENES],
            "test": [name for name in TEST_SCENES]
        }

        # ETH/UCY possibilities are the Cartesian product of these,
        # but note that some may not exist, such as ("eth", "train", "cyprus").
        # "*_loo" = Leave One Out (this is how the ETH/UCY dataset
        # is most commonly used).
        dataset_parts: List[Tuple[str, ...]] = [
            ("train", "val"),
            ("germany", "germany"),
        ]

        # Inverting the dict from above, associating every scene with its data split.
        scene_split_map: Dict[str, str] = {
            v_elem: k for k, v in scene_splits.items() for v_elem in v
        }

        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=KITTI_DT,
            parts=dataset_parts,
            scene_split_map=scene_split_map
        )
    

    ################################## DONE - NOT TESTED ################################## - FINISH EDITS AT WORK
    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)

        self.dataset_obj: Dict[str, pd.DataFrame] = dict()
        base_path = None # BASE PATH TO GLOBAL DATA HERE
        for scene_name in TRAIN_SCENES:
            data_filepath: Path = Path(self.metadata.data_dir) / (scene_name + ".txt") # READ PD FROM A CSV FILE - NEEDS "["frame_id", "track_id", "pos_x", "pos_y"]" format

            data = pd.read_csv(data_filepath, sep="\t", index_col=False, header=None)
            data.columns = ["frame_id", "track_id", "pos_x", "pos_y"]

            data["frame_id"] = pd.to_numeric(data["frame_id"], downcast="integer")
            data["frame_id"] = (data["frame_id"] - data["frame_id"].min()) // 10

            self.dataset_obj[scene_name] = data
            self.dataset_obj[scene_name + "_train"] = data


        # COPY AND PASTE FOR VAL AND TEST - IDK tHEY ONLY DO THAT FOR TRAIN AND VAL!!!
        ################# More data is set up in "get_agent_info()" - set up some of it here and for online all of it!!!


            # Creating a copy because we have to fix the frame_id values (to ensure they start from 0).
            self.dataset_obj[scene_name + "_val"] = data[
                data["frame_id"] >= TRAINVAL_FRAME_SPLITS[scene_name]
            ].copy()
            self.dataset_obj[scene_name + "_val"]["frame_id"] -= TRAINVAL_FRAME_SPLITS[
                scene_name
            ]






    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[NuscSceneRecord] = list()

        scenes_list: List[SceneMetadata] = list()
        for idx, scene_record in enumerate(self.dataset_obj.scene):
            scene_name: str = scene_record["name"]
            scene_desc: str = scene_record["description"].lower()
            scene_location: str = self.dataset_obj.get(
                "log", scene_record["log_token"]
            )["location"]
            scene_split: str = self.metadata.scene_split_map[scene_name]
            scene_length: int = scene_record["nbr_samples"]

            # Saving all scene records for later caching.
            all_scenes_list.append(
                NuscSceneRecord(
                    scene_name, scene_location, scene_length, scene_desc, idx
                )
            )

            if scene_location.split("-")[0] in scene_tag and scene_split in scene_tag:
                if scene_desc_contains is not None and not any(
                    desc_query in scene_desc for desc_query in scene_desc_contains
                ):
                    continue

                scene_metadata = SceneMetadata(
                    env_name=self.metadata.name,
                    name=scene_name,
                    dt=self.metadata.dt,
                    raw_data_idx=idx,
                )
                scenes_list.append(scene_metadata)

        self.cache_all_scenes_list(env_cache, all_scenes_list)
        return scenes_list

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        all_scenes_list: List[NuscSceneRecord] = env_cache.load_env_scenes_list(
            self.name
        )

        scenes_list: List[SceneMetadata] = list()
        for scene_record in all_scenes_list:
            (
                scene_name,
                scene_location,
                scene_length,
                scene_desc,
                data_idx,
            ) = scene_record
            scene_split: str = self.metadata.scene_split_map[scene_name]

            if scene_location.split("-")[0] in scene_tag and scene_split in scene_tag:
                if scene_desc_contains is not None and not any(
                    desc_query in scene_desc for desc_query in scene_desc_contains
                ):
                    continue

                scene_metadata = Scene(
                    self.metadata,
                    scene_name,
                    scene_location,
                    scene_split,
                    scene_length,
                    data_idx,
                    None,  # This isn't used if everything is already cached.
                    scene_desc,
                )
                scenes_list.append(scene_metadata)

        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, _, _, data_idx = scene_info

        scene_record = self.dataset_obj.scene[data_idx]
        scene_name: str = scene_record["name"]
        scene_desc: str = scene_record["description"].lower()
        scene_location: str = self.dataset_obj.get("log", scene_record["log_token"])[
            "location"
        ]
        scene_split: str = self.metadata.scene_split_map[scene_name]
        scene_length: int = scene_record["nbr_samples"]

        return Scene(
            self.metadata,
            scene_name,
            scene_location,
            scene_split,
            scene_length,
            data_idx,
            scene_record,
            scene_desc,
        )

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        ego_agent_info: AgentMetadata = AgentMetadata(
            name="ego",
            agent_type=AgentType.VEHICLE,
            first_timestep=0,
            last_timestep=scene.length_timesteps - 1,
            extent=FixedExtent(length=4.084, width=1.730, height=1.562),
        )

        agent_presence: List[List[AgentMetadata]] = [
            [ego_agent_info] for _ in range(scene.length_timesteps)
        ]

        agent_data_list: List[pd.DataFrame] = list()
        existing_agents: Dict[str, AgentMetadata] = dict()

        all_frames: List[Dict[str, Union[str, int]]] = list(
            nusc_utils.frame_iterator(self.dataset_obj, scene)
        )
        frame_idx_dict: Dict[str, int] = {
            frame_dict["token"]: idx for idx, frame_dict in enumerate(all_frames)
        }
        for frame_idx, frame_info in enumerate(all_frames):
            for agent_info in nusc_utils.agent_iterator(self.dataset_obj, frame_info):
                if agent_info["instance_token"] in existing_agents:
                    continue

                if not agent_info["next"]:
                    # There are some agents with only a single detection to them, we don't care about these.
                    continue

                agent: Agent = nusc_utils.agg_agent_data(
                    self.dataset_obj, agent_info, frame_idx, frame_idx_dict
                )

                for scene_ts in range(
                    agent.metadata.first_timestep, agent.metadata.last_timestep + 1
                ):
                    agent_presence[scene_ts].append(agent.metadata)

                existing_agents[agent.name] = agent.metadata

                agent_data_list.append(agent.data)

        ego_agent: Agent = nusc_utils.agg_ego_data(self.dataset_obj, scene)
        agent_data_list.append(ego_agent.data)

        agent_list: List[AgentMetadata] = [ego_agent_info] + list(
            existing_agents.values()
        )

        cache_class.save_agent_data(pd.concat(agent_data_list), cache_path, scene)

        return agent_list, agent_presence

    def cache_map(
        self,
        map_name: str,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        """
        No maps in this dataset!
        """
        pass

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        """
        No maps in this dataset!
        """
        pass
