from collections import OrderedDict
from typing import Sequence, Callable, Any

import torch
from torchrl.collectors import SyncDataCollector, WeightUpdaterBase
from torchrl.collectors.collectors import DEFAULT_EXPLORATION_TYPE
from torchrl.data import DEVICE_TYPING, Bounded
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs import (
    TransformedEnv,
    EnvBase,
    EnvCreator,
)
from torchrl.envs.utils import RandomPolicy
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.probabilistic import (
    InteractionType as ExplorationType,
    set_interaction_type as set_exploration_type,
)


class MultiPolicySyncDataCollector(SyncDataCollector):
    def __init__(
            self,
            create_env_fn: EnvBase | EnvCreator | Sequence[Callable[[], EnvBase]],
            defender_policy: None | (TensorDictModule | Callable[[TensorDictBase], TensorDictBase]) = None,
            attacker_policy: None | (TensorDictModule | Callable[[TensorDictBase], TensorDictBase]) = None,
            *,
            frames_per_batch: int,
            total_frames: int = -1,
            device: DEVICE_TYPING = None,
            storing_device: DEVICE_TYPING = None,
            policy_device: DEVICE_TYPING = None,
            env_device: DEVICE_TYPING = None,
            create_env_kwargs: dict[str, Any] | None = None,
            max_frames_per_traj: int | None = None,
            init_random_frames: int | None = None,
            reset_at_each_iter: bool = False,
            postproc: Callable[[TensorDictBase], TensorDictBase] | None = None,
            split_trajs: bool | None = None,
            exploration_type: ExplorationType = DEFAULT_EXPLORATION_TYPE,
            return_same_td: bool = False,
            reset_when_done: bool = True,
            interruptor=None,
            set_truncated: bool = False,
            use_buffers: bool | None = None,
            replay_buffer: ReplayBuffer | None = None,
            extend_buffer: bool = False,
            trust_policy: bool = None,
            compile_policy: bool | dict[str, Any] | None = None,
            cudagraph_policy: bool | dict[str, Any] | None = None,
            no_cuda_sync: bool = False,
            weight_updater: WeightUpdaterBase
                            | Callable[[], WeightUpdaterBase]
                            | None = None,
            **kwargs,
    ):
        # Initialize parent, but pass 'None' for the main policy argument
        # as we handle policies separately.
        super().__init__(
            create_env_fn=create_env_fn,
            policy=defender_policy,  # We set it so that specs are initialized
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            device=device,
            storing_device=storing_device,
            policy_device=policy_device,
            env_device=env_device,
            create_env_kwargs=create_env_kwargs,
            max_frames_per_traj=max_frames_per_traj,
            init_random_frames=init_random_frames,
            reset_at_each_iter=reset_at_each_iter,
            postproc=postproc,
            split_trajs=split_trajs,
            exploration_type=exploration_type,
            return_same_td=return_same_td,
            reset_when_done=reset_when_done,
            interruptor=interruptor,
            set_truncated=set_truncated,
            use_buffers=use_buffers,
            replay_buffer=replay_buffer,
            extend_buffer=extend_buffer,
            trust_policy=trust_policy,
            compile_policy=compile_policy,
            cudagraph_policy=cudagraph_policy,
            no_cuda_sync=no_cuda_sync,
            weight_updater=weight_updater,
            **kwargs,
        )

        # Store policies
        self.attacker_policy = attacker_policy
        self.defender_policy = defender_policy

        action_spec = self.env.full_action_spec["action"]
        modified_action_spec = Bounded(
            low=action_spec.space.low[..., 0, :],
            high=action_spec.space.high[..., 0, :],
            shape=torch.Size((*action_spec.shape[:-2], action_spec.shape[-1])),
            dtype=action_spec.dtype,
            device=action_spec.device,
        )
        if self.attacker_policy is None:
            self.attacker_policy = RandomPolicy(modified_action_spec)
        if self.defender_policy is None:
            self.defender_policy = RandomPolicy(modified_action_spec)

        # Move policies to the computation device if they exist and are modules
        if self.attacker_policy is not None and isinstance(self.attacker_policy, torch.nn.Module):
            self.attacker_policy.to(self.policy_device)
        if self.defender_policy is not None and isinstance(self.defender_policy, torch.nn.Module):
            self.defender_policy.to(self.policy_device)

    @torch.no_grad()
    def rollout(self) -> TensorDictBase:
        """Computes a rollout in the environment using the provided policies.

        Returns:
            TensorDictBase containing the computed rollout.
        """
        if self.reset_at_each_iter:
            self._shuttle.update(self.env.reset())

        # self._shuttle.fill_(("collector", "step_count"), 0)
        if self._use_buffers:
            self._final_rollout.fill_(("collector", "traj_ids"), -1)
        else:
            pass
        tensordicts = []
        with set_exploration_type(self.exploration_type):
            for t in range(self.frames_per_batch):
                if (
                        self.init_random_frames is not None
                        and self._frames < self.init_random_frames
                ):
                    self.env.rand_action(self._shuttle)
                    if (
                            self.policy_device is not None
                            and self.policy_device != self.env_device
                    ):
                        # TODO: This may break with exclusive / ragged lazy stacks
                        self._shuttle.apply(
                            lambda name, val: val.to(
                                device=self.policy_device, non_blocking=True
                            )
                            if name in self._policy_output_keys
                            else val,
                            out=self._shuttle,
                            named=True,
                            nested_keys=True,
                        )
                else:
                    if self._cast_to_policy_device:
                        if self.policy_device is not None:
                            # This is unsafe if the shuttle is in pin_memory -- otherwise cuda will be happy with non_blocking
                            non_blocking = (
                                    not self.no_cuda_sync
                                    or self.policy_device.type == "cuda"
                            )
                            policy_input = self._shuttle.to(
                                self.policy_device,
                                non_blocking=non_blocking,
                            )
                            if not self.no_cuda_sync:
                                self._sync_policy()
                        elif self.policy_device is None:
                            # we know the tensordict has a device otherwise we would not be here
                            # we can pass this, clear_device_ must have been called earlier
                            # policy_input = self._shuttle.clear_device_()
                            policy_input = self._shuttle
                    else:
                        policy_input = self._shuttle
                    # we still do the assignment for security
                    if self.compiled_policy:
                        raise NotImplementedError("compiled policy not supported yet")
                    defender_output = self.defender_policy(policy_input)
                    print(defender_output["logits"], defender_output["action"])
                    attacker_output = self.attacker_policy(policy_input)
                    print(attacker_output["logits"], attacker_output["action"])
                    print(defender_output["logits"], defender_output["action"])
                    combined_output = defender_output.clone()
                    combined_output["action"] = torch.stack(
                    [defender_output["action"], attacker_output["action"]], dim=-2
                    )
                    print(combined_output["action"])

                    if self._shuttle is not combined_output:
                        # ad-hoc update shuttle
                        self._shuttle.update(combined_output, keys_to_update=self._policy_output_keys)

                if self._cast_to_env_device:
                    if self.env_device is not None:
                        non_blocking = (
                                not self.no_cuda_sync or self.env_device.type == "cuda"
                        )
                        env_input = self._shuttle.to(
                            self.env_device, non_blocking=non_blocking
                        )
                        if not self.no_cuda_sync:
                            self._sync_env()
                    elif self.env_device is None:
                        # we know the tensordict has a device otherwise we would not be here
                        # we can pass this, clear_device_ must have been called earlier
                        # env_input = self._shuttle.clear_device_()
                        env_input = self._shuttle
                else:
                    env_input = self._shuttle
                env_output, env_next_output = self.env.step_and_maybe_reset(env_input)

                if self._shuttle is not env_output:
                    # ad-hoc update shuttle
                    next_data = env_output.get("next")
                    if self._shuttle_has_no_device:
                        # Make sure
                        next_data.clear_device_()
                    self._shuttle.set("next", next_data)

                if self.replay_buffer is not None and not self.extend_buffer:
                    self.replay_buffer.add(self._shuttle)
                    if self._increment_frames(self._shuttle.numel()):
                        return
                else:
                    if self.storing_device is not None:
                        non_blocking = (
                                not self.no_cuda_sync or self.storing_device.type == "cuda"
                        )
                        tensordicts.append(
                            self._shuttle.to(
                                self.storing_device, non_blocking=non_blocking
                            )
                        )
                        if not self.no_cuda_sync:
                            self._sync_storage()
                    else:
                        tensordicts.append(self._shuttle)

                # carry over collector data without messing up devices
                collector_data = self._shuttle.get("collector").copy()
                self._shuttle = env_next_output
                if self._shuttle_has_no_device:
                    self._shuttle.clear_device_()
                self._shuttle.set("collector", collector_data)
                self._update_traj_ids(env_output)

                if (
                        self.interruptor is not None
                        and self.interruptor.collection_stopped()
                ):
                    if self.replay_buffer is not None and not self.extend_buffer:
                        return
                    result = self._final_rollout
                    if self._use_buffers:
                        try:
                            torch.stack(
                                tensordicts,
                                self._final_rollout.ndim - 1,
                                out=self._final_rollout[..., : t + 1],
                            )
                        except RuntimeError:
                            with self._final_rollout.unlock_():
                                torch.stack(
                                    tensordicts,
                                    self._final_rollout.ndim - 1,
                                    out=self._final_rollout[..., : t + 1],
                                )
                    else:
                        result = TensorDict.maybe_dense_stack(tensordicts, dim=-1)
                    break
            else:
                if self._use_buffers:
                    result = self._final_rollout
                    try:
                        result = torch.stack(
                            tensordicts,
                            self._final_rollout.ndim - 1,
                            out=self._final_rollout,
                        )

                    except RuntimeError:
                        with self._final_rollout.unlock_():
                            result = torch.stack(
                                tensordicts,
                                self._final_rollout.ndim - 1,
                                out=self._final_rollout,
                            )
                elif self.replay_buffer is not None and not self.extend_buffer:
                    return
                else:
                    result = TensorDict.maybe_dense_stack(tensordicts, dim=-1)
                    result.refine_names(..., "time")

        return self._maybe_set_truncated(result)

    def state_dict(self) -> OrderedDict:
        """Returns the local state_dict of the data collector (environment and policy).

        Returns:
            an ordered dictionary with fields :obj:`"policy_state_dict"` and
            `"env_state_dict"`.

        """
        from torchrl.envs.batched_envs import BatchedEnvBase

        if isinstance(self.env, TransformedEnv):
            env_state_dict = self.env.transform.state_dict()
        elif isinstance(self.env, BatchedEnvBase):
            env_state_dict = self.env.state_dict()
        else:
            env_state_dict = OrderedDict()

        if hasattr(self.defender_policy, "state_dict") and hasattr(self.attacker_policy, "state_dict"):
            defender_policy_state_dict = self.defender_policy.state_dict()
            attacker_policy_state_dict = self.attacker_policy.state_dict()
            state_dict = OrderedDict(
                defender_policy_state_dict=defender_policy_state_dict,
                attacker_policy_state_dict=attacker_policy_state_dict,
                env_state_dict=env_state_dict,
            )
        elif hasattr(self.defender_policy, "state_dict"):
            defender_policy_state_dict = self.defender_policy.state_dict()
            state_dict = OrderedDict(
                defender_policy_state_dict=defender_policy_state_dict,
                env_state_dict=env_state_dict,
            )
        elif hasattr(self.attacker_policy, "state_dict"):
            attacker_policy_state_dict = self.attacker_policy.state_dict()
            state_dict = OrderedDict(
                attacker_policy_state_dict=attacker_policy_state_dict,
                env_state_dict=env_state_dict,
            )
        else:
            state_dict = OrderedDict(env_state_dict=env_state_dict)

        state_dict.update({"frames": self._frames, "iter": self._iter})

        return state_dict

    def load_state_dict(self, state_dict: OrderedDict, **kwargs) -> None:
        """Loads a state_dict on the environment and policy.

        Args:
            state_dict (OrderedDict): ordered dictionary containing the fields
                `"policy_state_dict"` and :obj:`"env_state_dict"`.

        """
        strict = kwargs.get("strict", True)
        if strict or "env_state_dict" in state_dict:
            self.env.load_state_dict(state_dict["env_state_dict"], **kwargs)
        if strict or "defender_policy_state_dict" in state_dict:
            self.defender_policy.load_state_dict(state_dict["defender_policy_state_dict"], **kwargs)
            self.policy.load_state_dict(state_dict["defender_policy_state_dict"], **kwargs)
        if strict or "attacker_policy_state_dict" in state_dict:
            self.attacker_policy.load_state_dict(state_dict["attacker_policy_state_dict"], **kwargs)
        self._frames = state_dict["frames"]
        self._iter = state_dict["iter"]

    def update_policy_weights_(
            self,
            policy_weights: TensorDictBase | None = None,
            *,
            worker_ids: int | list[int] | torch.device | list[torch.device] | None = None,
    ) -> None:
        raise NotImplementedError("update_policy_weights_ not implemented for MultiPolicySyncDataCollector")
