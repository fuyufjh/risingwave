// Copyright 2022 Singularity Data
// Licensed under the Apache License, Version 2.0 (the "License");
//
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

use std::collections::HashMap;
use std::mem::swap;
use std::sync::Arc;

use risingwave_common::error::ErrorCode::InternalError;
use risingwave_common::error::{ErrorCode, Result};
use risingwave_pb::plan::{TaskId as TaskIdProst, TaskOutputId as TaskOutputIdProst};
use tokio::sync::mpsc::{channel, Receiver, Sender};
use tokio::sync::{oneshot, RwLock};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

use super::stage::StageEvent;
use crate::scheduler::execution::query::QueryMessage::Stage;
use crate::scheduler::execution::query::QueryState::{Failed, Pending};
use crate::scheduler::execution::StageEvent::Scheduled;
use crate::scheduler::execution::{StageExecution, ROOT_TASK_ID, ROOT_TASK_OUTPUT_ID};
use crate::scheduler::plan_fragmenter::{Query, StageId};
use crate::scheduler::worker_node_manager::WorkerNodeManagerRef;
use crate::scheduler::{HummockSnapshotManagerRef, QueryResultFetcher};

/// Message sent to a `QueryRunner` to control its execution.
#[derive(Debug)]
pub enum QueryMessage {
    /// Commands to stop execution..
    Stop,

    /// Events passed running execution.
    Stage(StageEvent),
}

enum QueryState {
    /// Not scheduled yet.
    ///
    /// In this state, some data structures for starting executions are created to avoid holding
    /// them `QueryExecution`
    Pending {
        /// We create this runner before start execution to avoid hold unuseful fields in
        /// `QueryExecution`
        runner: QueryRunner,

        /// Receiver of root stage info.
        root_stage_receiver: oneshot::Receiver<QueryResultFetcher>,
    },

    /// Running
    Running {
        msg_sender: Sender<QueryMessage>,
        task_handle: JoinHandle<Result<()>>,
    },

    /// Failed
    Failed,

    /// Completed
    Completed,
}

pub struct QueryExecution {
    query: Arc<Query>,
    state: Arc<RwLock<QueryState>>,
    stage_executions: Arc<HashMap<StageId, Arc<StageExecution>>>,
}

struct QueryRunner {
    query: Arc<Query>,
    stage_executions: Arc<HashMap<StageId, Arc<StageExecution>>>,
    scheduled_stages_count: usize,
    // Query messages receiver. For example, stage state change events, query commands.
    msg_receiver: Receiver<QueryMessage>,
    // Sender of above message receiver. We need to keep it so that we can pass it to stages.
    msg_sender: Sender<QueryMessage>,

    // Will be set to `None` after all stage scheduled.
    root_stage_sender: Option<oneshot::Sender<QueryResultFetcher>>,

    epoch: u64,
    hummock_snapshot_manager: HummockSnapshotManagerRef,
}

impl QueryExecution {
    pub fn new(
        query: Query,
        epoch: u64,
        worker_node_manager: WorkerNodeManagerRef,
        hummock_snapshot_manager: HummockSnapshotManagerRef,
    ) -> Self {
        let query = Arc::new(query);
        let (sender, receiver) = channel(100);

        let stage_executions = {
            let mut stage_executions: HashMap<StageId, Arc<StageExecution>> =
                HashMap::with_capacity(query.stage_graph.stages.len());

            for stage_id in query.stage_graph.stage_ids_by_topo_order() {
                let children_stages = query
                    .stage_graph
                    .get_child_stages_unchecked(&stage_id)
                    .iter()
                    .map(|s| stage_executions[s].clone())
                    .collect::<Vec<Arc<StageExecution>>>();

                let stage_exec = Arc::new(StageExecution::new(
                    epoch,
                    query.stage_graph.stages[&stage_id].clone(),
                    worker_node_manager.clone(),
                    sender.clone(),
                    children_stages,
                ));
                stage_executions.insert(stage_id, stage_exec);
            }
            Arc::new(stage_executions)
        };

        let (root_stage_sender, root_stage_receiver) = oneshot::channel::<QueryResultFetcher>();

        let runner = QueryRunner {
            query: query.clone(),
            stage_executions: stage_executions.clone(),
            msg_receiver: receiver,
            root_stage_sender: Some(root_stage_sender),
            msg_sender: sender,
            scheduled_stages_count: 0,

            epoch,
            hummock_snapshot_manager,
        };

        let state = Pending {
            runner,
            root_stage_receiver,
        };

        Self {
            query,
            state: Arc::new(RwLock::new(state)),
            stage_executions,
        }
    }

    /// Start execution of this query.
    pub async fn start(&self) -> Result<QueryResultFetcher> {
        let mut state = self.state.write().await;
        let mut cur_state = Failed;
        swap(&mut *state, &mut cur_state);

        match cur_state {
            QueryState::Pending {
                runner,
                root_stage_receiver,
            } => {
                let msg_sender = runner.msg_sender.clone();
                let task_handle = tokio::spawn(async move {
                    let query_id = runner.query.query_id.clone();
                    runner.run().await.map_err(|e| {
                        error!("Query {:?} failed, reason: {:?}", query_id, e);
                        e
                    })
                });

                let root_stage = root_stage_receiver.await.map_err(|e| {
                    InternalError(format!("Starting query execution failed: {:?}", e))
                })?;

                info!(
                    "Received root stage query result fetcher: {:?}, query id: {:?}",
                    root_stage, self.query.query_id
                );

                *state = QueryState::Running {
                    msg_sender,
                    task_handle,
                };

                Ok(root_stage)
            }
            s => {
                // Restore old state
                *state = s;
                Err(ErrorCode::InternalError("Query not pending!".to_string()).into())
            }
        }
    }

    /// Cancel execution of this query.
    pub async fn abort(&mut self) -> Result<()> {
        todo!()
    }
}

impl QueryRunner {
    async fn run(mut self) -> Result<()> {
        // Start leaf stages.
        for stage_id in &self.query.leaf_stages() {
            // TODO: We should not return error here, we should abort query.
            info!(
                "Starting query stage: {:?}-{:?}",
                self.query.query_id, stage_id
            );
            self.stage_executions
                .get(stage_id)
                .as_ref()
                .unwrap()
                .start()
                .await
                .map_err(|e| {
                    error!("Failed to start stage: {}, reason: {:?}", stage_id, e);
                    e
                })?;
            info!(
                "Query stage {:?}-{:?} started.",
                self.query.query_id, stage_id
            );
        }

        // Schedule stages when leaf stages all scheduled
        while let Some(msg) = self.msg_receiver.recv().await {
            match msg {
                Stage(Scheduled(stage_id)) => {
                    info!(
                        "Query stage {:?}-{:?} scheduled.",
                        self.query.query_id, stage_id
                    );
                    self.scheduled_stages_count += 1;

                    if self.scheduled_stages_count == self.stage_executions.len() {
                        // Now all stages schedules, send root stage info.
                        self.send_root_stage_info().await;
                    } else {
                        for parent in self.query.get_parents(&stage_id) {
                            if self.all_children_scheduled(parent).await {
                                self.get_stage_execution_unchecked(parent)
                                    .start()
                                    .await
                                    .map_err(|e| {
                                        error!(
                                            "Failed to start stage: {}, reason: {:?}",
                                            stage_id, e
                                        );
                                        e
                                    })?;
                            }
                        }
                    }
                }
                _ => {
                    return Err(ErrorCode::NotImplemented(
                        "unsupported type for QueryRunner.run".to_string(),
                        None.into(),
                    )
                    .into())
                }
            }
        }

        info!("Query runner {:?} finished.", self.query.query_id);
        Ok(())
    }

    async fn send_root_stage_info(&mut self) {
        let root_task_status = self.stage_executions[&self.query.root_stage_id()]
            .get_task_status_unchecked(ROOT_TASK_ID);

        let root_task_output_id = {
            let root_task_id_prost = TaskIdProst {
                query_id: self.query.query_id.clone().id,
                stage_id: self.query.root_stage_id(),
                task_id: ROOT_TASK_ID,
            };

            TaskOutputIdProst {
                task_id: Some(root_task_id_prost),
                output_id: ROOT_TASK_OUTPUT_ID,
            }
        };

        let root_stage_result = QueryResultFetcher::new(
            self.epoch,
            self.hummock_snapshot_manager.clone(),
            root_task_output_id,
            root_task_status.task_host_unchecked(),
        );

        // Consume sender here.
        let mut tmp_sender = None;
        swap(&mut self.root_stage_sender, &mut tmp_sender);

        if let Err(e) = tmp_sender.unwrap().send(root_stage_result) {
            warn!("Query execution dropped: {:?}", e);
        } else {
            debug!("Root stage for {:?} sent.", self.query.query_id);
        }
    }

    async fn all_children_scheduled(&self, stage_id: &StageId) -> bool {
        for child in self.query.stage_graph.get_child_stages_unchecked(stage_id) {
            if !self
                .get_stage_execution_unchecked(child)
                .is_scheduled()
                .await
            {
                return false;
            }
        }
        true
    }

    fn get_stage_execution_unchecked(&self, stage_id: &StageId) -> Arc<StageExecution> {
        self.stage_executions.get(stage_id).unwrap().clone()
    }
}
