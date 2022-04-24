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

use std::sync::Arc;

use futures_async_stream::try_stream;
use itertools::Itertools;
use risingwave_common::array::column::Column;
use risingwave_common::array::DataChunk;
use risingwave_common::catalog::{Field, Schema};
use risingwave_common::error::{ErrorCode, Result, RwError};
use risingwave_expr::expr::{build_from_prost, BoxedExpression};
use risingwave_expr::vector_op::agg::{
    create_sorted_grouper, AggStateFactory, BoxedAggState, BoxedSortedGrouper, EqGroups,
};
use risingwave_pb::plan::plan_node::NodeBody;

use crate::executor::ExecutorBuilder;
use crate::executor2::{BoxedDataChunkStream, BoxedExecutor2, BoxedExecutor2Builder, Executor2};

/// `SortAggExecutor` implements the sort aggregate algorithm, where tuples
/// belonging to the same group are continuous because they are sorted by the
/// group columns.
///
/// As a special case, simple aggregate without groups satisfies the requirement
/// automatically because all tuples should be aggregated together.
pub struct SortAggExecutor2 {
    agg_states: Vec<BoxedAggState>,
    group_exprs: Vec<BoxedExpression>,
    sorted_groupers: Vec<BoxedSortedGrouper>,
    child: BoxedExecutor2,
    schema: Schema,
    identity: String,
}

impl BoxedExecutor2Builder for SortAggExecutor2 {
    fn new_boxed_executor2(source: &ExecutorBuilder) -> Result<BoxedExecutor2> {
        ensure!(source.plan_node().get_children().len() == 1);
        let proto_child = source
            .plan_node()
            .get_children()
            .get(0)
            .ok_or_else(|| ErrorCode::InternalError(String::from("")))?;
        let child = source.clone_for_plan(proto_child).build2()?;

        let sort_agg_node = try_match_expand!(
            source.plan_node().get_node_body().unwrap(),
            NodeBody::SortAgg
        )?;

        let agg_states = sort_agg_node
            .get_agg_calls()
            .iter()
            .map(|x| AggStateFactory::new(x)?.create_agg_state())
            .collect::<Result<Vec<BoxedAggState>>>()?;

        let group_exprs = sort_agg_node
            .get_group_keys()
            .iter()
            .map(build_from_prost)
            .collect::<Result<Vec<BoxedExpression>>>()?;

        let sorted_groupers = group_exprs
            .iter()
            .map(|e| create_sorted_grouper(e.return_type()))
            .collect::<Result<Vec<BoxedSortedGrouper>>>()?;

        let fields = group_exprs
            .iter()
            .map(|e| e.return_type())
            .chain(agg_states.iter().map(|e| e.return_type()))
            .map(Field::unnamed)
            .collect::<Vec<Field>>();

        Ok(Box::new(Self {
            agg_states,
            group_exprs,
            sorted_groupers,
            child,
            schema: Schema { fields },
            identity: source.plan_node().get_identity().clone(),
        }))
    }
}

impl Executor2 for SortAggExecutor2 {
    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn identity(&self) -> &str {
        &self.identity
    }

    fn execute(self: Box<Self>) -> BoxedDataChunkStream {
        self.do_execute()
    }
}

impl SortAggExecutor2 {
    #[try_stream(boxed, ok = DataChunk, error = RwError)]
    async fn do_execute(mut self: Box<Self>) {
        let cardinality = 1;
        let mut group_builders = self
            .group_exprs
            .iter()
            .map(|e| e.return_type().create_array_builder(cardinality))
            .collect::<Result<Vec<_>>>()?;
        let mut array_builders = self
            .agg_states
            .iter()
            .map(|e| e.return_type().create_array_builder(cardinality))
            .collect::<Result<Vec<_>>>()?;

        #[for_await]
        for child_chunk in self.child.execute() {
            let child_chunk = child_chunk?;
            let group_arrays = self
                .group_exprs
                .iter_mut()
                .map(|expr| expr.eval(&child_chunk))
                .collect::<Result<Vec<_>>>()?;

            let groups = self
                .sorted_groupers
                .iter()
                .zip_eq(&group_arrays)
                .map(|(grouper, array)| grouper.split_groups(array))
                .collect::<Result<Vec<EqGroups>>>()?;
            let groups = EqGroups::intersect(&groups);

            self.sorted_groupers
                .iter_mut()
                .zip_eq(&group_arrays)
                .zip_eq(&mut group_builders)
                .try_for_each(|((grouper, array), builder)| {
                    grouper.update_and_output_with_sorted_groups(array, builder, &groups)
                })?;

            self.agg_states
                .iter_mut()
                .zip_eq(&mut array_builders)
                .try_for_each(|(state, builder)| {
                    state.update_and_output_with_sorted_groups(&child_chunk, builder, &groups)
                })?;
        }

        self.sorted_groupers
            .iter()
            .zip_eq(&mut group_builders)
            .try_for_each(|(grouper, builder)| grouper.output(builder))?;
        self.agg_states
            .iter()
            .zip_eq(&mut array_builders)
            .try_for_each(|(state, builder)| state.output(builder))?;

        let columns = group_builders
            .into_iter()
            .chain(array_builders)
            .map(|b| Ok(Column::new(Arc::new(b.finish()?))))
            .collect::<Result<Vec<_>>>()?;

        let output = DataChunk::builder().columns(columns).build();
        yield output;
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use futures::StreamExt;
    use risingwave_common::array::{Array as _, I32Array, I64Array};
    use risingwave_common::array_nonnull;
    use risingwave_common::catalog::{Field, Schema};
    use risingwave_common::types::DataType;
    use risingwave_expr::expr::build_from_prost;
    use risingwave_pb::data::data_type::TypeName;
    use risingwave_pb::data::DataType as ProstDataType;
    use risingwave_pb::expr::agg_call::{Arg, Type};
    use risingwave_pb::expr::expr_node::RexNode;
    use risingwave_pb::expr::expr_node::Type::InputRef;
    use risingwave_pb::expr::{AggCall, ExprNode, InputRefExpr};

    use super::*;
    use crate::executor::test_utils::MockExecutor;

    #[tokio::test]
    #[allow(clippy::many_single_char_names)]
    async fn execute_sum_int32() -> Result<()> {
        let a = Arc::new(array_nonnull! { I32Array, [1, 2, 3] }.into());
        let chunk = DataChunk::builder().columns(vec![Column::new(a)]).build();
        let schema = Schema {
            fields: vec![Field::unnamed(DataType::Int32)],
        };
        let mut child = MockExecutor::new(schema);
        child.add(chunk);

        let prost = AggCall {
            r#type: Type::Sum as i32,
            args: vec![Arg {
                input: Some(InputRefExpr { column_idx: 0 }),
                r#type: Some(ProstDataType {
                    type_name: TypeName::Int32 as i32,
                    ..Default::default()
                }),
            }],
            return_type: Some(ProstDataType {
                type_name: TypeName::Int64 as i32,
                ..Default::default()
            }),
            distinct: false,
        };

        let s = AggStateFactory::new(&prost)?.create_agg_state()?;

        let group_exprs: Vec<BoxedExpression> = vec![];
        let agg_states = vec![s];
        let fields = group_exprs
            .iter()
            .map(|e| e.return_type())
            .chain(agg_states.iter().map(|e| e.return_type()))
            .map(Field::unnamed)
            .collect::<Vec<Field>>();
        let executor = Box::new(SortAggExecutor2 {
            agg_states,
            group_exprs: vec![],
            sorted_groupers: vec![],
            child: Box::new(child),
            schema: Schema { fields },
            identity: "SortAggExecutor".to_string(),
        });

        let mut stream = executor.execute();
        let res = stream.next().await.unwrap();
        assert_matches!(res, Ok(_));
        assert_matches!(stream.next().await, None);

        let actual = res?.column_at(0).array();
        let actual: &I64Array = actual.as_ref().into();
        let v = actual.iter().collect::<Vec<Option<i64>>>();
        assert_eq!(v, vec![Some(6)]);

        Ok(())
    }

    #[tokio::test]
    #[allow(clippy::many_single_char_names)]
    async fn execute_sum_int32_grouped() -> Result<()> {
        use risingwave_common::array::ArrayImpl;
        let a: Arc<ArrayImpl> = Arc::new(array_nonnull! { I32Array, [1, 2, 3] }.into());
        let chunk = DataChunk::builder()
            .columns(vec![
                Column::new(a.clone()),
                Column::new(Arc::new(array_nonnull! { I32Array, [1, 1, 3] }.into())),
                Column::new(Arc::new(array_nonnull! { I32Array, [7, 8, 8] }.into())),
            ])
            .build();
        let schema = Schema {
            fields: vec![
                Field::unnamed(DataType::Int32),
                Field::unnamed(DataType::Int32),
                Field::unnamed(DataType::Int32),
            ],
        };
        let mut child = MockExecutor::new(schema);
        child.add(chunk);
        let chunk = DataChunk::builder()
            .columns(vec![
                Column::new(a),
                Column::new(Arc::new(array_nonnull! { I32Array, [3, 4, 4] }.into())),
                Column::new(Arc::new(array_nonnull! { I32Array, [8, 8, 8] }.into())),
            ])
            .build();
        child.add(chunk);

        let prost = AggCall {
            r#type: Type::Sum as i32,
            args: vec![Arg {
                input: Some(InputRefExpr { column_idx: 0 }),
                r#type: Some(ProstDataType {
                    type_name: TypeName::Int32 as i32,
                    ..Default::default()
                }),
            }],
            return_type: Some(ProstDataType {
                type_name: TypeName::Int64 as i32,
                ..Default::default()
            }),
            distinct: false,
        };

        let s = AggStateFactory::new(&prost)?.create_agg_state()?;

        let group_exprs = (1..=2)
            .map(|idx| {
                build_from_prost(&ExprNode {
                    expr_type: InputRef as i32,
                    return_type: Some(ProstDataType {
                        type_name: TypeName::Int32 as i32,
                        ..Default::default()
                    }),
                    rex_node: Some(RexNode::InputRef(InputRefExpr { column_idx: idx })),
                })
            })
            .collect::<Result<Vec<BoxedExpression>>>()?;
        let sorted_groupers = group_exprs
            .iter()
            .map(|e| create_sorted_grouper(e.return_type()))
            .collect::<Result<Vec<BoxedSortedGrouper>>>()?;

        let agg_states = vec![s];
        let fields = group_exprs
            .iter()
            .map(|e| e.return_type())
            .chain(agg_states.iter().map(|e| e.return_type()))
            .map(Field::unnamed)
            .collect::<Vec<Field>>();

        let executor = Box::new(SortAggExecutor2 {
            agg_states,
            group_exprs,
            sorted_groupers,
            child: Box::new(child),
            schema: Schema { fields },
            identity: "SortAggExecutor".to_string(),
        });

        let fields = &executor.schema().fields;
        assert_eq!(fields[0].data_type, DataType::Int32);
        assert_eq!(fields[1].data_type, DataType::Int32);
        assert_eq!(fields[2].data_type, DataType::Int64);

        let mut stream = executor.execute();
        let res = stream.next().await.unwrap();
        assert_matches!(res, Ok(_));
        assert_matches!(stream.next().await, None);

        let chunk = res?;
        let actual = chunk.column_at(2).array();
        let actual: &I64Array = actual.as_ref().into();
        let v = actual.iter().collect::<Vec<Option<i64>>>();
        assert_eq!(v, vec![Some(1), Some(2), Some(4), Some(5)]);

        assert_eq!(
            chunk
                .column_at(0)
                .array()
                .as_int32()
                .iter()
                .collect::<Vec<_>>(),
            vec![Some(1), Some(1), Some(3), Some(4)]
        );
        assert_eq!(
            chunk
                .column_at(1)
                .array()
                .as_int32()
                .iter()
                .collect::<Vec<_>>(),
            vec![Some(7), Some(8), Some(8), Some(8)]
        );

        Ok(())
    }
}
