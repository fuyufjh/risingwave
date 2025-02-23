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

use fixedbitset::FixedBitSet;
use risingwave_common::error::Result;

use crate::binder::BoundQuery;
use crate::optimizer::plan_node::{LogicalLimit, LogicalTopN};
use crate::optimizer::property::{Distribution, Order};
use crate::optimizer::PlanRoot;
use crate::planner::Planner;

pub const LIMIT_ALL_COUNT: usize = usize::MAX / 2;

impl Planner {
    /// Plan a [`BoundQuery`]. Need to bind before planning.
    pub fn plan_query(&mut self, query: BoundQuery) -> Result<PlanRoot> {
        let mut plan = self.plan_set_expr(query.body)?;
        let order = Order {
            field_order: query.order,
        };
        if query.limit.is_some() || query.offset.is_some() {
            let limit = query.limit.unwrap_or(LIMIT_ALL_COUNT);
            let offset = query.offset.unwrap_or_default();
            plan = if order.field_order.is_empty() {
                // Create a logical limit if with limit/offset but without order-by
                LogicalLimit::create(plan, limit, offset)
            } else {
                // Create a logical top-n if with limit/offset and order-by
                LogicalTopN::create(plan, limit, offset, order.clone())
            }
        }
        let dist = Distribution::Single;
        let mut out_fields = FixedBitSet::with_capacity(plan.schema().len());
        out_fields.insert_range(..);
        let root = PlanRoot::new(plan, dist, order, out_fields);
        Ok(root)
    }
}
