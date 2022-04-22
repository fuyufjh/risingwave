// Copyright 2022 Singularity Data
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::hash_map::{Entry, HashMap};
use std::collections::{BTreeMap, HashSet};
use std::ops::Deref;
use std::sync::Arc;

use assert_matches::assert_matches;
use itertools::Itertools;
use risingwave_common::catalog::TableId;
use risingwave_common::error::Result;
use risingwave_pb::plan::TableRefId;
use risingwave_pb::stream_plan::stream_node::Node;
use risingwave_pb::stream_plan::{
    BatchParallelInfo, Dispatcher, DispatcherType, MergeNode, StreamActor, StreamNode,
};
use risingwave_pb::ProstFieldNotFound;

use crate::model::{ActorId, LocalActorId, LocalFragmentId};
use crate::storage::MetaStore;
use crate::stream::{CreateMaterializedViewContext, FragmentManagerRef};

/// A list of actors with order.
#[derive(Debug, Clone)]
pub struct OrderedActorLink(pub Vec<LocalActorId>);

impl OrderedActorLink {
    pub fn to_global_ids(&self, actor_id_offset: u32, actor_id_len: u32) -> Self {
        Self(
            self.0
                .iter()
                .map(|x| x.to_global_id(actor_id_offset, actor_id_len))
                .collect(),
        )
    }

    pub fn as_global_ids(&self) -> Vec<u32> {
        Self::slice_as_global_ids(self.0.as_slice())
    }

    pub fn slice_as_global_ids(data: &[LocalActorId]) -> Vec<u32> {
        data.iter().map(|x| x.as_global_id()).collect()
    }
}

pub struct StreamActorDownstream {
    /// Dispatcher
    /// TODO: refactor to `DispatchStrategy`.
    dispatcher: Dispatcher,

    /// Downstream actors.
    actors: OrderedActorLink,

    /// Whether to place the downstream actors on the same node
    same_worker_node: bool,
}

pub struct StreamActorUpstream {
    /// Upstream actors
    actors: OrderedActorLink,

    /// Whether to place the upstream actors on the same node
    same_worker_node: bool,
}

/// [`StreamActorBuilder`] builds a stream actor in a stream DAG.
pub struct StreamActorBuilder {
    /// actor id field
    actor_id: LocalActorId,

    /// associated fragment id
    fragment_id: LocalFragmentId,

    /// associated stream node
    nodes: Arc<StreamNode>,

    /// downstream dispatchers (dispatcher, downstream actor, hash mapping)
    downstreams: Vec<StreamActorDownstream>,

    /// upstreams, exchange node operator_id -> upstream actor ids
    upstreams: HashMap<u64, StreamActorUpstream>,

    /// whether this actor builder has been sealed
    sealed: bool,

    /// parallel degree of corresponding fragment
    parallel_degree: u32,
}

impl StreamActorBuilder {
    pub fn new(
        actor_id: LocalActorId,
        fragment_id: LocalFragmentId,
        node: Arc<StreamNode>,
        parallel_degree: u32,
    ) -> Self {
        Self {
            actor_id,
            fragment_id,
            nodes: node,
            downstreams: vec![],
            upstreams: HashMap::new(),
            sealed: false,
            parallel_degree,
        }
    }

    pub fn get_id(&self) -> LocalActorId {
        self.actor_id
    }

    pub fn get_fragment_id(&self) -> LocalFragmentId {
        self.fragment_id
    }

    pub fn get_parallel_degree(&self) -> u32 {
        self.parallel_degree
    }

    /// Add a dispatcher to this actor. Note that the `downstream_actor_id` field must be left
    /// empty, as we will fill this out when building actors.
    pub fn add_dispatcher(
        &mut self,
        dispatcher: Dispatcher,
        downstream_actors: OrderedActorLink,
        same_worker_node: bool,
    ) {
        assert!(!self.sealed);
        // TODO: we should have a non-proto Dispatcher type here
        assert!(
            dispatcher.downstream_actor_id.is_empty(),
            "should leave downstream_actor_id empty, will be filled later"
        );
        assert!(
            dispatcher.hash_mapping.is_none(),
            "should leave hash_mapping empty, will be filled later"
        );
        self.downstreams.push(StreamActorDownstream {
            dispatcher,
            actors: downstream_actors,
            same_worker_node,
        });
    }

    /// Build an actor from given information. At the same time, convert local actor id to global
    /// actor id.
    pub fn seal(&mut self, actor_id_offset: u32, actor_id_len: u32) {
        assert!(!self.sealed);

        self.actor_id = self.actor_id.to_global_id(actor_id_offset, actor_id_len);
        self.downstreams = std::mem::take(&mut self.downstreams)
            .into_iter()
            .map(
                |StreamActorDownstream {
                     dispatcher,
                     actors: downstreams,
                     same_worker_node,
                 }| {
                    let downstreams = downstreams.to_global_ids(actor_id_offset, actor_id_len);

                    if dispatcher.r#type == DispatcherType::NoShuffle as i32 {
                        assert_eq!(
                            downstreams.0.len(),
                            1,
                            "no shuffle should only have one actor downstream"
                        );
                        assert!(
                            dispatcher.column_indices.is_empty(),
                            "should leave `column_indices` empty"
                        );
                    }

                    StreamActorDownstream {
                        dispatcher,
                        actors: downstreams,
                        same_worker_node,
                    }
                },
            )
            .collect();

        self.upstreams = std::mem::take(&mut self.upstreams)
            .into_iter()
            .map(
                |(
                    exchange_id,
                    StreamActorUpstream {
                        actors,
                        same_worker_node,
                    },
                )| {
                    (
                        exchange_id,
                        StreamActorUpstream {
                            actors: actors.to_global_ids(actor_id_offset, actor_id_len),
                            same_worker_node,
                        },
                    )
                },
            )
            .collect();
        self.sealed = true;
    }

    /// Build an actor after seal.
    pub fn build(&self) -> StreamActor {
        assert!(self.sealed);

        let mut dispatcher = self
            .downstreams
            .iter()
            .map(
                |StreamActorDownstream {
                     dispatcher, actors, ..
                 }| Dispatcher {
                    downstream_actor_id: actors.as_global_ids(),
                    ..dispatcher.clone()
                },
            )
            .collect_vec();

        // If there's no dispatcher, add an empty broadcast. TODO: Can be removed later.
        if dispatcher.is_empty() {
            dispatcher = vec![Dispatcher {
                r#type: DispatcherType::Broadcast.into(),
                ..Default::default()
            }]
        }

        StreamActor {
            actor_id: self.actor_id.as_global_id(),
            fragment_id: self.fragment_id.as_global_id(),
            nodes: Some(self.nodes.deref().clone()),
            dispatcher,
            upstream_actor_id: self
                .upstreams
                .iter()
                .flat_map(|(_, StreamActorUpstream { actors, .. })| actors.0.iter().copied())
                .map(|x| x.as_global_id())
                .collect(), // TODO: store each upstream separately
            same_worker_node_as_upstream: self.upstreams.iter().any(
                |(
                    _,
                    StreamActorUpstream {
                        same_worker_node, ..
                    },
                )| *same_worker_node,
            ),
        }
    }
}

/// [`StreamGraphBuilder`] build a stream graph. It injects some information to achieve
/// dependencies. See `build_inner` for more details.
pub struct StreamGraphBuilder<S> {
    actor_builders: BTreeMap<LocalActorId, StreamActorBuilder>,
    /// (ctx) fragment manager.
    fragment_manager: FragmentManagerRef<S>,
}

impl<S> StreamGraphBuilder<S>
where
    S: MetaStore,
{
    pub fn new(fragment_manager: FragmentManagerRef<S>) -> Self {
        Self {
            actor_builders: BTreeMap::new(),
            fragment_manager,
        }
    }

    /// Insert new generated actor.
    pub fn add_actor(&mut self, actor: StreamActorBuilder) {
        self.actor_builders.insert(actor.get_id(), actor);
    }

    /// Number of actors in the graph builder
    pub fn actor_len(&self) -> usize {
        self.actor_builders.len()
    }

    /// Add dependency between two connected node in the graph. Only provide `mapping` when it's
    /// hash mapping.
    pub fn add_link(
        &mut self,
        upstream_actor_ids: &[LocalActorId],
        downstream_actor_ids: &[LocalActorId],
        exchange_operator_id: u64,
        dispatcher: Dispatcher,
        same_worker_node: bool,
    ) {
        if dispatcher.get_type().unwrap() == DispatcherType::NoShuffle {
            assert_eq!(
                upstream_actor_ids.len(),
                downstream_actor_ids.len(),
                "mismatched length when procssing no-shuffle exchange: {:?} -> {:?} on exchange {}",
                upstream_actor_ids,
                downstream_actor_ids,
                exchange_operator_id
            );

            // update 1v1 relationship
            upstream_actor_ids
                .iter()
                .zip_eq(downstream_actor_ids.iter())
                .for_each(|(upstream_id, downstream_id)| {
                    self.actor_builders
                        .get_mut(upstream_id)
                        .unwrap()
                        .add_dispatcher(
                            dispatcher.clone(),
                            OrderedActorLink(vec![*downstream_id]),
                            same_worker_node,
                        );

                    let ret = self
                        .actor_builders
                        .get_mut(downstream_id)
                        .unwrap()
                        .upstreams
                        .insert(
                            exchange_operator_id,
                            StreamActorUpstream {
                                actors: OrderedActorLink(vec![*upstream_id]),
                                same_worker_node,
                            },
                        );

                    assert!(
                        ret.is_none(),
                        "duplicated exchange input {} for no-shuffle actors {:?} -> {:?}",
                        exchange_operator_id,
                        upstream_id,
                        downstream_id
                    );
                });

            return;
        }

        // otherwise, make m * n links between actors.

        assert!(
            !same_worker_node,
            "same_worker_node only applies to 1v1 dispatchers."
        );

        // update actors to have dispatchers, link upstream -> downstream.
        upstream_actor_ids.iter().for_each(|upstream_id| {
            self.actor_builders
                .get_mut(upstream_id)
                .unwrap()
                .add_dispatcher(
                    dispatcher.clone(),
                    OrderedActorLink(downstream_actor_ids.to_vec()),
                    same_worker_node,
                );
        });

        // update actors to have upstreams, link downstream <- upstream.
        downstream_actor_ids.iter().for_each(|downstream_id| {
            let ret = self
                .actor_builders
                .get_mut(downstream_id)
                .unwrap()
                .upstreams
                .insert(
                    exchange_operator_id,
                    StreamActorUpstream {
                        actors: OrderedActorLink(upstream_actor_ids.to_vec()),
                        same_worker_node,
                    },
                );
            assert!(
                ret.is_none(),
                "duplicated exchange input {} for actors {:?} -> {:?}",
                exchange_operator_id,
                upstream_actor_ids,
                downstream_actor_ids
            );
        });
    }

    /// Build final stream DAG with dependencies with current actor builders.
    pub fn build(
        &mut self,
        ctx: &mut CreateMaterializedViewContext,
        actor_id_offset: u32,
        actor_id_len: u32,
        table_ids_map: &HashMap<u64, u32>,
        start_table_id: u32,
    ) -> Result<HashMap<LocalFragmentId, Vec<StreamActor>>> {
        let mut graph = HashMap::new();

        for builder in self.actor_builders.values_mut() {
            builder.seal(actor_id_offset, actor_id_len);
        }

        for builder in self.actor_builders.values() {
            let actor_id = builder.actor_id;
            let mut actor = builder.build();
            let mut dispatch_upstreams = vec![];
            let mut upstream_actors = builder
                .upstreams
                .iter()
                .map(|(id, StreamActorUpstream { actors, .. })| (*id, actors.clone()))
                .collect();

            actor.nodes = Some(self.build_inner(
                ctx,
                &mut dispatch_upstreams,
                actor.get_nodes()?,
                &mut upstream_actors,
                builder.get_fragment_id(),
                builder.get_parallel_degree(),
                table_ids_map,
                start_table_id,
            )?);

            graph
                .entry(builder.get_fragment_id())
                .or_insert(vec![])
                .push(actor);

            for up_id in dispatch_upstreams {
                match ctx.dispatches.entry(up_id) {
                    Entry::Occupied(mut o) => {
                        o.get_mut().push(actor_id.as_global_id());
                    }
                    Entry::Vacant(v) => {
                        v.insert(vec![actor_id.as_global_id()]);
                    }
                }
            }
        }
        for actor_ids in ctx.upstream_node_actors.values_mut() {
            actor_ids.sort_unstable();
            actor_ids.dedup();
        }

        Ok(graph)
    }

    /// Build stream actor inside, two works will be done:
    /// 1. replace node's input with [`MergeNode`] if it is `ExchangeNode`, and swallow
    /// mergeNode's input.
    /// 2. ignore root node when it's `ExchangeNode`.
    /// 3. replace node's `ExchangeNode` input with [`MergeNode`] and resolve its upstream actor
    /// ids if it is a `ChainNode`.
    #[allow(clippy::too_many_arguments)]
    pub fn build_inner(
        &self,
        ctx: &mut CreateMaterializedViewContext,
        dispatch_upstreams: &mut Vec<ActorId>,
        stream_node: &StreamNode,
        upstream_actor_id: &mut HashMap<u64, OrderedActorLink>,
        fragment_id: LocalFragmentId,
        parallel_degree: u32,
        table_ids_map: &HashMap<u64, u32>,
        start_table_id: u32,
    ) -> Result<StreamNode> {
        match stream_node.get_node()? {
            Node::ExchangeNode(_) => {
                panic!("ExchangeNode should be eliminated from the top of the plan node when converting fragments to actors")
            }
            Node::ChainNode(_) => self.resolve_chain_node(
                ctx,
                dispatch_upstreams,
                stream_node,
                fragment_id,
                parallel_degree,
            ),
            _ => {
                let mut new_stream_node = stream_node.clone();
                if let Node::HashJoinNode(node) = new_stream_node
                    .node
                    .as_mut()
                    .ok_or(ProstFieldNotFound("prost stream node field not found"))?
                {
                    // The operator id must be assigned with table ids. Otherwise it is a logic
                    // error.
                    let left_table_id =
                        table_ids_map.get(&new_stream_node.operator_id).unwrap() + start_table_id;
                    let right_table_id = left_table_id + 1;
                    node.left_table_ref_id = Some(TableRefId {
                        table_id: left_table_id as i32,
                        ..Default::default()
                    });
                    node.right_table_ref_id = Some(TableRefId {
                        table_id: right_table_id as i32,
                        ..Default::default()
                    });
                }
                for (idx, input) in stream_node.input.iter().enumerate() {
                    match input.get_node()? {
                        Node::ExchangeNode(_) => {
                            assert!(!input.get_fields().is_empty());
                            new_stream_node.input[idx] = StreamNode {
                                input: vec![],
                                pk_indices: input.pk_indices.clone(),
                                node: Some(Node::MergeNode(MergeNode {
                                    upstream_actor_id: upstream_actor_id
                                        .remove(&input.get_operator_id())
                                        .expect("failed to find upstream actor id for given exchange node").as_global_ids(),
                                    fields: input.get_fields().clone(),
                                })),
                                fields: input.get_fields().clone(),
                                operator_id: input.operator_id,
                                identity: "MergeExecutor".to_string(),
                                append_only: input.append_only,
                            };
                        }
                        Node::ChainNode(_) => {
                            new_stream_node.input[idx] = self.resolve_chain_node(
                                ctx,
                                dispatch_upstreams,
                                input,
                                fragment_id,
                                parallel_degree,
                            )?;
                        }
                        _ => {
                            new_stream_node.input[idx] = self.build_inner(
                                ctx,
                                dispatch_upstreams,
                                input,
                                upstream_actor_id,
                                fragment_id,
                                parallel_degree,
                                table_ids_map,
                                start_table_id,
                            )?;
                        }
                    }
                }
                Ok(new_stream_node)
            }
        }
    }

    fn resolve_chain_node(
        &self,
        ctx: &mut CreateMaterializedViewContext,
        dispatch_upstreams: &mut Vec<ActorId>,
        stream_node: &StreamNode,
        fragment_id: LocalFragmentId,
        parallel_degree: u32,
    ) -> Result<StreamNode> {
        if let Node::ChainNode(chain_node) = stream_node.get_node().unwrap() {
            let input = stream_node.get_input();
            assert_eq!(input.len(), 2);
            let table_id = TableId::from(&chain_node.table_ref_id);

            let (assigned_upstream, parallel_index) =
                self.assign_upstream_for_chain(table_id, ctx, fragment_id)?;

            dispatch_upstreams.extend(assigned_upstream.iter());
            let chain_upstream_table_node_actors = self
                .fragment_manager
                .blocking_table_node_actors(&table_id)?;
            let chain_upstream_node_actors = chain_upstream_table_node_actors
                .iter()
                .flat_map(|(node_id, actor_ids)| {
                    actor_ids.iter().map(|actor_id| (*node_id, *actor_id))
                })
                .filter(|(_, actor_id)| assigned_upstream.contains(actor_id))
                .into_group_map();
            for (node_id, actor_ids) in chain_upstream_node_actors {
                match ctx.upstream_node_actors.entry(node_id) {
                    Entry::Occupied(mut o) => {
                        o.get_mut().extend(actor_ids.iter());
                    }
                    Entry::Vacant(v) => {
                        v.insert(actor_ids);
                    }
                }
            }

            let merge_node = &input[0];
            assert_matches!(merge_node.node, Some(Node::MergeNode(_)));

            let mut batch_plan_node = input[1].clone();
            if let Some(Node::BatchPlanNode(ref mut node)) = batch_plan_node.node {
                node.parallel_info = Some(BatchParallelInfo {
                    degree: parallel_degree,
                    index: parallel_index,
                });
            } else {
                unreachable!("input[1].node should be a BatchPlanNode");
            }

            let chain_input = vec![
                StreamNode {
                    input: vec![],
                    pk_indices: stream_node.pk_indices.clone(),
                    node: Some(Node::MergeNode(MergeNode {
                        upstream_actor_id: Vec::from_iter(assigned_upstream.into_iter()),
                        fields: chain_node.upstream_fields.clone(),
                    })),
                    fields: chain_node.upstream_fields.clone(),
                    operator_id: merge_node.operator_id,
                    identity: "MergeExecutor".to_string(),
                    append_only: stream_node.append_only,
                },
                batch_plan_node,
            ];

            Ok(StreamNode {
                input: chain_input,
                pk_indices: stream_node.pk_indices.clone(),
                node: Some(Node::ChainNode(chain_node.clone())),
                operator_id: stream_node.operator_id,
                identity: "ChainExecutor".to_string(),
                fields: chain_node.upstream_fields.clone(),
                append_only: stream_node.append_only,
            })
        } else {
            unreachable!()
        }
    }

    /// maintain a `<fragment_id, hashset<upstream_actor_id>>` mapping
    /// For each actor beginning with chain, we will assign *one* upstream table sink actor
    /// to this actor. (assert chain actor's parallel degree == upstream parallel degree).
    fn assign_upstream_for_chain(
        &self,
        table_id: TableId,
        ctx: &mut CreateMaterializedViewContext,
        fragment_id: LocalFragmentId,
    ) -> Result<(HashSet<ActorId>, u32)> {
        let remaining = match ctx
            .chain_upstream_assignment
            .entry(fragment_id.as_global_id())
        {
            Entry::Vacant(v) => {
                let mut upstream_actor_ids = match ctx.table_sink_map.entry(table_id) {
                    Entry::Vacant(v) => {
                        let actor_ids = self
                            .fragment_manager
                            .blocking_get_table_sink_actor_ids(&table_id)?;
                        v.insert(actor_ids).clone()
                    }
                    Entry::Occupied(o) => o.get().clone(),
                };
                upstream_actor_ids.dedup();
                v.insert(upstream_actor_ids)
            }
            Entry::Occupied(o) => o.into_mut(),
        };

        // choose one upstream from remaining.
        assert!(!remaining.is_empty());

        // TODO: remove this when we deprecate Java frontend.
        let (assigned_upstreams, parallel_index) = if ctx.is_legacy_frontend {
            (HashSet::<ActorId>::from_iter(remaining.iter().cloned()), 0)
        } else {
            let upstream_id = remaining.pop().unwrap();
            (
                HashSet::<ActorId>::from([upstream_id]),
                remaining.len() as u32,
            )
        };

        Ok((assigned_upstreams, parallel_index))
    }
}
