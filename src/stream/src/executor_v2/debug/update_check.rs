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

use std::iter::once;
use std::sync::Arc;

use futures_async_stream::try_stream;
use itertools::Itertools;
use risingwave_common::array::Op;

use crate::executor::Message;
use crate::executor_v2::error::StreamExecutorError;
use crate::executor_v2::{ExecutorInfo, MessageStream};

/// Streams wrapped by `update_check` will check whether the two rows of updates are next to each
/// other.
#[try_stream(ok = Message, error = StreamExecutorError)]
pub async fn update_check(info: Arc<ExecutorInfo>, input: impl MessageStream) {
    #[for_await]
    for message in input {
        let message = message?;

        if let Message::Chunk(chunk) = &message {
            for ((op1, row1), (op2, row2)) in once(None)
                .chain(chunk.rows().map(Some))
                .chain(once(None))
                .map(|r| (r.unzip()))
                .tuple_windows()
            {
                if (op1 == None && op2 == Some(Op::UpdateInsert)) // the first row is U+
                    || (op1 == Some(Op::UpdateDelete) && op2 != Some(Op::UpdateInsert))
                {
                    panic!(
                        "update check failed on `{}`: expect U+ after  U-:\n first row: {:?}\nsecond row: {:?}",
                        info.identity,
                        row1,
                        row2,
                    )
                }
            }
        }

        yield message;
    }
}

#[cfg(test)]
mod tests {
    use std::iter::once;

    use futures::{pin_mut, StreamExt};
    use risingwave_common::array::StreamChunk;

    use super::*;
    use crate::executor_v2::test_utils::MockSource;
    use crate::executor_v2::Executor;

    #[should_panic]
    #[tokio::test]
    async fn test_not_next_to_each_other() {
        let chunk = StreamChunk::from_str(
            "     I
            U-  114 
            U-  514
            U+ 1919 
            U+  810",
        );

        let mut source = MockSource::new(Default::default(), vec![]);
        source.push_chunks(once(chunk));

        let checked = update_check(source.info().into(), source.boxed().execute());
        pin_mut!(checked);

        checked.next().await.unwrap().unwrap(); // should panic
    }

    #[should_panic]
    #[tokio::test]
    async fn test_first_one_update_insert() {
        let chunk = StreamChunk::from_str(
            "     I
            U+  114",
        );

        let mut source = MockSource::new(Default::default(), vec![]);
        source.push_chunks(once(chunk));

        let checked = update_check(source.info().into(), source.boxed().execute());
        pin_mut!(checked);

        checked.next().await.unwrap().unwrap(); // should panic
    }

    #[should_panic]
    #[tokio::test]
    async fn test_last_one_update_delete() {
        let chunk = StreamChunk::from_str(
            "        I
            U-     114 
            U+     514
            U- 1919810",
        );

        let mut source = MockSource::new(Default::default(), vec![]);
        source.push_chunks(once(chunk));

        let checked = update_check(source.info().into(), source.boxed().execute());
        pin_mut!(checked);

        checked.next().await.unwrap().unwrap(); // should panic
    }

    #[tokio::test]
    async fn test_empty_chunk() {
        let chunk = StreamChunk::default();

        let mut source = MockSource::new(Default::default(), vec![]);
        source.push_chunks(once(chunk));

        let checked = update_check(source.info().into(), source.boxed().execute());
        pin_mut!(checked);

        checked.next().await.unwrap().unwrap();
    }
}
