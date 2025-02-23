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
use std::vec;

use futures_async_stream::try_stream;
use itertools::Itertools;
use risingwave_common::array::column::Column;
use risingwave_common::array::{DataChunk, I32Array};
use risingwave_common::catalog::{Field, Schema};
use risingwave_common::error::{Result, RwError};
use risingwave_common::util::chunk_coalesce::DEFAULT_CHUNK_BUFFER_SIZE;
use risingwave_expr::expr::{build_from_prost, BoxedExpression};
use risingwave_pb::plan::plan_node::NodeBody;

use crate::executor::ExecutorBuilder;
use crate::executor2::{BoxedDataChunkStream, BoxedExecutor2, BoxedExecutor2Builder, Executor2};

/// `ValuesExecutor` implements Values executor.
pub struct ValuesExecutor2 {
    rows: vec::IntoIter<Vec<BoxedExpression>>,
    schema: Schema,
    identity: String,
    chunk_size: usize,
}

impl ValuesExecutor2 {
    pub(crate) fn new(
        rows: Vec<Vec<BoxedExpression>>,
        schema: Schema,
        identity: String,
        chunk_size: usize,
    ) -> Self {
        Self {
            rows: rows.into_iter(),
            schema,
            identity,
            chunk_size,
        }
    }
}

impl Executor2 for ValuesExecutor2 {
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

impl ValuesExecutor2 {
    #[try_stream(boxed, ok = DataChunk, error = RwError)]
    async fn do_execute(mut self: Box<Self>) {
        if !self.rows.is_empty() {
            let cardinality = self.rows.len();
            ensure!(cardinality > 0);

            if self.schema.fields.is_empty() {
                let cardinality = self.rows.len();
                self.rows = vec![].into_iter();
                yield DataChunk::new_dummy(cardinality);
            } else {
                while !self.rows.is_empty() {
                    let one_row_array = I32Array::from_slice(&[Some(1)])?;
                    // We need a one row chunk rather than an empty chunk because constant
                    // expression's eval result is same size as input chunk
                    // cardinality.
                    let one_row_chunk = DataChunk::builder()
                        .columns(vec![Column::new(Arc::new(one_row_array.into()))])
                        .build();

                    let chunk_size = self.chunk_size.min(self.rows.len());
                    let mut array_builders = self.schema.create_array_builders(chunk_size)?;
                    for row in self.rows.by_ref().take(chunk_size) {
                        for (expr, builder) in row.into_iter().zip_eq(&mut array_builders) {
                            let out = expr.eval(&one_row_chunk)?;
                            builder.append_array(&out)?;
                        }
                    }

                    let columns = array_builders
                        .into_iter()
                        .map(|builder| builder.finish().map(|arr| Column::new(Arc::new(arr))))
                        .collect::<Result<Vec<Column>>>()?;

                    let chunk = DataChunk::builder().columns(columns).build();

                    yield chunk
                }
            }
        }
    }
}
impl BoxedExecutor2Builder for ValuesExecutor2 {
    fn new_boxed_executor2(source: &ExecutorBuilder) -> Result<BoxedExecutor2> {
        let value_node = try_match_expand!(
            source.plan_node().get_node_body().unwrap(),
            NodeBody::Values
        )?;

        let mut rows: Vec<Vec<BoxedExpression>> = Vec::with_capacity(value_node.get_tuples().len());
        for row in value_node.get_tuples() {
            let expr_row = row
                .get_cells()
                .iter()
                .map(build_from_prost)
                .collect::<Result<Vec<BoxedExpression>>>()?;
            rows.push(expr_row);
        }

        let fields = value_node
            .get_fields()
            .iter()
            .map(Field::from)
            .collect::<Vec<Field>>();

        Ok(Box::new(Self {
            rows: rows.into_iter(),
            schema: Schema { fields },
            identity: source.plan_node().get_identity().clone(),
            chunk_size: DEFAULT_CHUNK_BUFFER_SIZE,
        }))
    }
}

#[cfg(test)]
mod tests {
    use futures::stream::StreamExt;
    use risingwave_common::array;
    use risingwave_common::array::{I16Array, I32Array, I64Array};
    use risingwave_common::catalog::{Field, Schema};
    use risingwave_common::types::{DataType, ScalarImpl};
    use risingwave_expr::expr::{BoxedExpression, LiteralExpression};

    use crate::executor2::{Executor2, ValuesExecutor2};

    #[tokio::test]
    async fn test_values_executor() {
        let exprs = vec![
            Box::new(LiteralExpression::new(
                DataType::Int16,
                Some(ScalarImpl::Int16(1)),
            )) as BoxedExpression,
            Box::new(LiteralExpression::new(
                DataType::Int32,
                Some(ScalarImpl::Int32(2)),
            )),
            Box::new(LiteralExpression::new(
                DataType::Int64,
                Some(ScalarImpl::Int64(3)),
            )),
        ];

        let fields = exprs
            .iter() // for each column
            .map(|col| Field::unnamed(col.return_type()))
            .collect::<Vec<Field>>();

        let values_executor = Box::new(ValuesExecutor2 {
            rows: vec![exprs].into_iter(),
            schema: Schema { fields },
            identity: "ValuesExecutor2".to_string(),
            chunk_size: 1024,
        });

        let fields = &values_executor.schema().fields;
        assert_eq!(fields[0].data_type, DataType::Int16);
        assert_eq!(fields[1].data_type, DataType::Int32);
        assert_eq!(fields[2].data_type, DataType::Int64);

        let mut stream = values_executor.execute();
        let result = stream.next().await.unwrap();

        if let Ok(result) = result {
            assert_eq!(
                *result.column_at(0).array(),
                array! {I16Array, [Some(1_i16)]}.into()
            );
            assert_eq!(
                *result.column_at(1).array(),
                array! {I32Array, [Some(2)]}.into()
            );
            assert_eq!(
                *result.column_at(2).array(),
                array! {I64Array, [Some(3)]}.into()
            );
        }
    }

    #[tokio::test]
    async fn test_chunk_split_size() {
        let rows = [
            Box::new(LiteralExpression::new(
                DataType::Int32,
                Some(ScalarImpl::Int32(1)),
            )) as BoxedExpression,
            Box::new(LiteralExpression::new(
                DataType::Int32,
                Some(ScalarImpl::Int32(2)),
            )) as BoxedExpression,
            Box::new(LiteralExpression::new(
                DataType::Int32,
                Some(ScalarImpl::Int32(3)),
            )) as BoxedExpression,
            Box::new(LiteralExpression::new(
                DataType::Int32,
                Some(ScalarImpl::Int32(4)),
            )) as BoxedExpression,
        ]
        .into_iter()
        .map(|expr| vec![expr])
        .collect::<Vec<_>>();

        let fields = vec![Field::unnamed(DataType::Int32)];

        let values_executor = Box::new(ValuesExecutor2::new(
            rows,
            Schema { fields },
            "ValuesExecutor2".to_string(),
            3,
        ));
        let mut stream = values_executor.execute();
        assert_eq!(stream.next().await.unwrap().unwrap().cardinality(), 3);
        assert_eq!(stream.next().await.unwrap().unwrap().cardinality(), 1);
        assert!(stream.next().await.is_none());
    }

    // Handle the possible case of ValuesNode([[]])
    #[tokio::test]
    async fn test_no_column_values_executor() {
        let values_executor = Box::new(ValuesExecutor2::new(
            vec![vec![]],
            Schema::default(),
            "ValuesExecutor2".to_string(),
            1024,
        ));
        let mut stream = values_executor.execute();

        let result = stream.next().await.unwrap().unwrap();
        assert_eq!(result.cardinality(), 1);
        assert_eq!(result.dimension(), 0);

        assert!(stream.next().await.is_none());
    }
}
