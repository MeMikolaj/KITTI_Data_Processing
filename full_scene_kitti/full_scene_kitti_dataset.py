from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Type

import pandas as pd
import numpy as np
import os
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
from trajdata.dataset_specific.scene_records import FullSceneKittiSceneRecord
from trajdata.utils import arr_utils

SCENES: Final[List[str]] = [
    #"0001",
    "0000",
    #"0020",
    #"0002",
    #"0003",
    #"0004",
    #"0005",
    #"0006",
    #"0018",
    #"0020"
]


KITTI_DT: Final[float] = 0.05

class FullSceneKittiDataset(RawDataset):

    ################################## DONE & TESTED ##################################
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        scene_splits: Dict[str, List[str]] = {
            "test": [name for name in SCENES],
        }

        dataset_parts: List[Tuple[str, ...]] = [
            ("test",),
            ("poland",),
        ]

        # Inverting the dict from above, associating every scene with its data split.
        scene_split_map: Dict[str, str] = {
            v_elem: k for k, v in scene_splits.items() for v_elem in v
        }

                                                                                            # scene_split_map is correct
                                                                                            # dataset_parts - correct-ish

        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=KITTI_DT,
            parts=dataset_parts,
            scene_split_map=scene_split_map
        )
    

    ################################## DONE & TESTED ##################################
    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)

        self.dataset_obj: Dict[str, pd.DataFrame] = dict()
        base_path = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/raw' # BASE PATH TO GLOBAL DATA HERE
        
        # "TEST" DATA
        for scene_name in SCENES:
            
            scene_path = os.path.join(base_path, scene_name, (scene_name + '.csv'))
            df_input = pd.read_csv(scene_path, index_col=False)

            data = df_input[['frame_ID', 'object_ID', 'x', 'y']].rename(
                columns={
                    'frame_ID': 'frame_id',
                    'object_ID': 'track_id',
                    'x': 'pos_x',
                    'y': 'pos_y'
                }
            )
            
            data["frame_id"] = pd.to_numeric(data["frame_id"], downcast="integer")
            data["frame_id"] = (data["frame_id"] - data["frame_id"].min())
            self.dataset_obj[scene_name] = data


    ################################## DONE & TESTED ##################################
    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[FullSceneKittiSceneRecord] = list()

        scenes_list: List[SceneMetadata] = list()
        for idx, (scene_name, scene_df) in enumerate(self.dataset_obj.items()):
            scene_location: str = "poland"

            # print("...... ...... ...... Scene split map:")
            # print(self.metadata.scene_split_map)
            
            scene_split: str = self.metadata.scene_split_map[scene_name]
            scene_length: int = scene_df["frame_id"].max().item() + 1

            # Saving all scene records for later caching.
            all_scenes_list.append(
                FullSceneKittiSceneRecord(scene_name, scene_location, scene_length, scene_split, idx)
            )
            # if (
            #     scene_location in scene_tag
            #     and scene_split in scene_tag
            #     and scene_desc_contains is None
            # ):
            scene_metadata = SceneMetadata(
                env_name=self.metadata.name,
                name=scene_name,
                dt=self.metadata.dt,
                raw_data_idx=idx,
            )
            scenes_list.append(scene_metadata)

        self.cache_all_scenes_list(env_cache, all_scenes_list)
        print("_get_matching_scenes_from_obj FINISHED")
        return scenes_list


    ################################## DONE & TESTED ##################################
    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        all_scenes_list: List[FullSceneKittiSceneRecord] = env_cache.load_env_scenes_list(self.name)

        scenes_list: List[Scene] = list()
        for scene_record in all_scenes_list:
            (
                scene_name,
                scene_location,
                scene_length,
                scene_split,
                data_idx,
            ) = scene_record

            # if (
            #     scene_location in scene_tag
            #     and scene_split in scene_tag
            #     and scene_desc_contains is None
            # ):
            scene_metadata = Scene(
                self.metadata,
                scene_name,
                scene_location,
                scene_split,
                scene_length,
                data_idx,
                None,  # This isn't used if everything is already cached.
            )
            scenes_list.append(scene_metadata)
        print("_get_matching_scenes_from_cache FINISHED")
        return scenes_list

    ################################## DONE & TESTED ##################################
    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, scene_name, _, data_idx = scene_info

        scene_data: pd.DataFrame = self.dataset_obj[scene_name]
        scene_location: str = "poland"
        scene_split: str = self.metadata.scene_split_map[scene_name]
        scene_length: int = scene_data["frame_id"].max().item() + 1

        return Scene(
            self.metadata,
            scene_name,
            scene_location,
            scene_split,
            scene_length,
            data_idx,
            None,  # No data access info necessary for the ETH/UCY/Kitti? datasets.
        )
        
        
    ################################## DONE & TESTED ##################################
    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        scene_data: pd.DataFrame = self.dataset_obj[scene.name].copy()
        scene_data.rename(
            columns={
                "frame_id": "scene_ts",
                "track_id": "agent_id",
                "pos_x": "x",
                "pos_y": "y",
            },
            inplace=True,
        )

        scene_data["agent_id"] = pd.to_numeric(
            scene_data["agent_id"], downcast="integer"
        )

        scene_data.set_index(["agent_id", "scene_ts"], inplace=True)
        scene_data.sort_index(inplace=True)
        scene_data.reset_index(level=1, inplace=True)

        agent_ids: np.ndarray = scene_data.index.get_level_values(0).to_numpy()

        # Add in zero for z value
        scene_data["z"] = np.zeros_like(scene_data["x"])

        ### Calculating agent velocities
        scene_data[["vx", "vy"]] = (
            arr_utils.agent_aware_diff(scene_data[["x", "y"]].to_numpy(), agent_ids)
            / KITTI_DT
        )

        ### Calculating agent accelerations
        scene_data[["ax", "ay"]] = (
            arr_utils.agent_aware_diff(scene_data[["vx", "vy"]].to_numpy(), agent_ids)
            / KITTI_DT
        )

        # This is likely to be very noisy... Unfortunately, ETH/UCY only
        # provide center of mass data.
        scene_data["heading"] = np.arctan2(scene_data["vy"], scene_data["vx"])

        agent_list: List[AgentMetadata] = list()
        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(scene.length_timesteps)
        ]
        agents_to_remove: List[int] = list()
        for agent_id, frames in scene_data.groupby("agent_id")["scene_ts"]:
            if frames.shape[0] <= 1:
                # There are some agents with only a single detection to them, we don't care about these.
                agents_to_remove.append(agent_id)
                continue

            start_frame: int = frames.iat[0].item()
            last_frame: int = frames.iat[-1].item()
            
            # print(f"Last Frame num: {last_frame}")
            # print(f"Start Frame num: {start_frame}")
            # print(f"Frames shape is: {frames.shape[0]}")
            # print(f"So the shape is: {frames.shape[0]} < {last_frame - start_frame + 1}.")
            # print(f"Scene name is: {scene.name}")
            # print(f"Agent ID is: {agent_id}")
            # print("-----")
            if frames.shape[0] < last_frame - start_frame + 1:
                raise ValueError(f"Kitti has missing frames. Scene: {scene.name}, Agent: {agent_id}")

            agent_metadata = AgentMetadata(
                name=str(agent_id),
                agent_type=AgentType.VEHICLE,
                first_timestep=start_frame,
                last_timestep=last_frame,
                # These values are as ballpark as it gets...
                extent=FixedExtent(length=4.084, width=1.730, height=1.562), # Avg car dim
            )

            agent_list.append(agent_metadata)
            for frame in frames:
                agent_presence[frame].append(agent_metadata)

        # Removing agents with only one detection.
        scene_data.drop(index=agents_to_remove, inplace=True)

        # Changing the agent_id dtype to str
        scene_data.reset_index(inplace=True)
        scene_data["agent_id"] = scene_data["agent_id"].astype(str)
        scene_data.set_index(["agent_id", "scene_ts"], inplace=True)

        cache_class.save_agent_data(
            scene_data,
            cache_path,
            scene,
        )

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
