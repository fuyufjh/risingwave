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

use std::fmt;

use risingwave_pb::plan::plan_node::NodeBody;
use risingwave_pb::plan::{ExchangeNode, MergeSortExchangeNode};

use super::{PlanBase, PlanRef, PlanTreeNodeUnary, ToBatchProst, ToDistributedBatch};
use crate::optimizer::property::{Distribution, Order};

/// `BatchExchange` imposes a particular distribution on its input
/// without changing its content.
#[derive(Debug, Clone)]
pub struct BatchExchange {
    pub base: PlanBase,
    input: PlanRef,
}

impl BatchExchange {
    pub fn new(input: PlanRef, order: Order, dist: Distribution) -> Self {
        let ctx = input.ctx();
        let schema = input.schema().clone();
        let _pk_indices = input.pk_indices().to_vec();
        let base = PlanBase::new_batch(ctx, schema, dist, order);
        BatchExchange { base, input }
    }
}

impl fmt::Display for BatchExchange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "BatchExchange {{ order: {}, dist: {:?} }}",
            self.base.order, self.base.dist
        )
    }
}

impl PlanTreeNodeUnary for BatchExchange {
    fn input(&self) -> PlanRef {
        self.input.clone()
    }

    fn clone_with_input(&self, input: PlanRef) -> Self {
        Self::new(input, self.order().clone(), self.distribution().clone())
    }
}
impl_plan_tree_node_for_unary! {BatchExchange}

impl ToDistributedBatch for BatchExchange {
    fn to_distributed(&self) -> PlanRef {
        unreachable!()
    }
}

/// The serialization of Batch Exchange is default cuz it will be rewritten in scheduler.
impl ToBatchProst for BatchExchange {
    fn to_batch_prost_body(&self) -> NodeBody {
        if self.base.order.is_any() {
            NodeBody::Exchange(ExchangeNode {
                sources: vec![],
                input_schema: self.base.schema.to_prost(),
            })
        } else {
            NodeBody::MergeSortExchange(MergeSortExchangeNode {
                exchange_node: Some(ExchangeNode {
                    sources: vec![],
                    input_schema: self.base.schema.to_prost(),
                }),
                column_orders: self.base.order.to_protobuf(&self.base.schema),
            })
        }
    }
}
