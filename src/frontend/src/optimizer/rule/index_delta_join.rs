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

use std::collections::HashSet;
use std::rc::Rc;

use itertools::Itertools;
use risingwave_pb::plan::JoinType;

use super::super::plan_node::*;
use super::{BoxedRule, Rule};

/// Use index scan and delta joins for supported queries.
pub struct IndexDeltaJoinRule {}

impl Rule for IndexDeltaJoinRule {
    fn apply(&self, plan: PlanRef) -> Option<PlanRef> {
        let join = plan.as_stream_hash_join()?;
        if join.eq_join_predicate().has_non_eq() || join.join_type() != JoinType::Inner {
            return Some(plan);
        }

        fn match_through_exchange(plan: PlanRef) -> Option<PlanRef> {
            if let Some(exchange) = plan.as_stream_exchange() {
                match_through_exchange(exchange.input())
            } else if plan.as_stream_table_scan().is_some() {
                Some(plan)
            } else {
                None
            }
        }

        let input_left_dyn = match_through_exchange(Rc::clone(&join.inputs()[0]))?;
        let input_left = input_left_dyn.as_stream_table_scan()?;
        let input_right_dyn = match_through_exchange(Rc::clone(&join.inputs()[1]))?;
        let input_right = input_right_dyn.as_stream_table_scan()?;
        let left_indices = join.eq_join_predicate().left_eq_indexes();
        let right_indices = join.eq_join_predicate().right_eq_indexes();

        fn match_indexes(join_indices: &[usize], table_scan: &StreamTableScan) -> bool {
            if table_scan.logical().indexes().is_empty() {
                return false;
            }
            let column_descs = table_scan.logical().column_descs();
            // We assume column id of create index's MV is exactly the same as corresponding columns
            // in the original table. We generate a list of column ids, and match them against
            // prefixes of indexes.
            let columns_to_match = join_indices
                .iter()
                .map(|idx| column_descs[*idx].column_id)
                .collect_vec();
            for index in table_scan.logical().indexes() {
                // A HashSet containing remaining columns to match
                let mut remaining_to_match =
                    columns_to_match.iter().copied().collect::<HashSet<_>>();

                // Begin match join columns with index prefix. e.g., if the join columns are `a, b,
                // c`, and the index has `a, b, c` or `a, c, b` or any combination as prefix, then
                // we can use this index.
                for ordered_column in &index.pk {
                    let column_id = ordered_column.column_desc.column_id;

                    match remaining_to_match.remove(&column_id) {
                        true => continue, // matched
                        false => break,   // not matched
                    }
                }

                if remaining_to_match.is_empty() {
                    return true;
                }
            }

            false
        }

        if match_indexes(&left_indices, input_left) && match_indexes(&right_indices, input_right) {
            // TODO: rewrite child to stream index scan
            Some(join.to_delta_join().into())
        } else {
            Some(plan)
        }
    }
}

impl IndexDeltaJoinRule {
    pub fn create() -> BoxedRule {
        Box::new(Self {})
    }
}
