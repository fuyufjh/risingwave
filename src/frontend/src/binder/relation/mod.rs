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

use std::collections::hash_map::Entry;
use std::str::FromStr;

use risingwave_common::catalog::DEFAULT_SCHEMA_NAME;
use risingwave_common::error::{ErrorCode, Result};
use risingwave_common::types::DataType;
use risingwave_sqlparser::ast::{ObjectName, TableAlias, TableFactor};

use super::bind_context::ColumnBinding;
use crate::binder::Binder;

mod join;
mod subquery;
mod table_or_source;
mod window_table_function;
pub use join::BoundJoin;
pub use subquery::BoundSubquery;
pub use table_or_source::{BoundBaseTable, BoundSource, BoundTableSource};
pub use window_table_function::{BoundWindowTableFunction, WindowTableFunctionKind};

/// A validated item that refers to a table-like entity, including base table, subquery, join, etc.
/// It is usually part of the `from` clause.
#[derive(Debug)]
pub enum Relation {
    Source(Box<BoundSource>),
    BaseTable(Box<BoundBaseTable>),
    Subquery(Box<BoundSubquery>),
    Join(Box<BoundJoin>),
    WindowTableFunction(Box<BoundWindowTableFunction>),
}

impl Binder {
    /// return the (`schema_name`, `table_name`)
    pub fn resolve_table_name(name: ObjectName) -> Result<(String, String)> {
        let mut identifiers = name.0;
        let table_name = identifiers
            .pop()
            .ok_or_else(|| ErrorCode::InternalError("empty table name".into()))?
            .value;

        let schema_name = identifiers
            .pop()
            .map(|ident| ident.value)
            .unwrap_or_else(|| DEFAULT_SCHEMA_NAME.into());

        Ok((schema_name, table_name))
    }

    /// Fill the [`BindContext`](super::BindContext) for table.
    pub(super) fn bind_context(
        &mut self,
        columns: impl IntoIterator<Item = (String, DataType, bool)>,
        table_name: String,
        alias: Option<TableAlias>,
    ) -> Result<()> {
        let (table_name, column_aliases) = match alias {
            None => (table_name, vec![]),
            Some(TableAlias { name, columns }) => (name.value, columns),
        };

        let begin = self.context.columns.len();
        // Column aliases can be less than columns, but not more.
        // It also needs to skip hidden columns.
        let mut alias_iter = column_aliases.into_iter().fuse();
        columns
            .into_iter()
            .enumerate()
            .for_each(|(index, (name, data_type, is_hidden))| {
                let name = match is_hidden {
                    true => name,
                    false => alias_iter.next().map(|t| t.value).unwrap_or(name),
                };
                self.context.columns.push(ColumnBinding::new(
                    table_name.clone(),
                    name.clone(),
                    begin + index,
                    data_type,
                    is_hidden,
                ));
                self.context
                    .indexs_of
                    .entry(name)
                    .or_default()
                    .push(self.context.columns.len() - 1);
            });
        if alias_iter.next().is_some() {
            return Err(ErrorCode::BindError(format!(
                "table \"{table_name}\" has less columns available but more aliases specified",
            ))
            .into());
        }

        match self.context.range_of.entry(table_name.clone()) {
            Entry::Occupied(_) => Err(ErrorCode::InternalError(format!(
                "Duplicated table name while binding context: {}",
                table_name
            ))
            .into()),
            Entry::Vacant(entry) => {
                entry.insert((begin, self.context.columns.len()));
                Ok(())
            }
        }
    }

    pub(super) fn bind_table_factor(&mut self, table_factor: TableFactor) -> Result<Relation> {
        match table_factor {
            TableFactor::Table { name, alias, args } => {
                if args.is_empty() {
                    let (schema_name, table_name) = Self::resolve_table_name(name)?;
                    self.bind_table_or_source(&schema_name, &table_name, alias)
                } else {
                    let kind =
                        WindowTableFunctionKind::from_str(&name.0[0].value).map_err(|_| {
                            ErrorCode::NotImplemented(
                                format!("unknown window function kind: {}", name.0[0].value),
                                1191.into(),
                            )
                        })?;
                    Ok(Relation::WindowTableFunction(Box::new(
                        self.bind_window_table_function(kind, args)?,
                    )))
                }
            }
            TableFactor::Derived {
                lateral,
                subquery,
                alias,
            } => {
                if lateral {
                    Err(ErrorCode::NotImplemented("unsupported lateral".into(), None.into()).into())
                } else {
                    Ok(Relation::Subquery(Box::new(
                        self.bind_subquery_relation(*subquery, alias)?,
                    )))
                }
            }
            _ => Err(ErrorCode::NotImplemented(
                format!("unsupported table factor {:?}", table_factor),
                None.into(),
            )
            .into()),
        }
    }
}
