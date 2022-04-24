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

use std::collections::BTreeMap;
use std::ops::RangeBounds;

use super::shared_buffer_batch::SharedBufferBatch;
use crate::hummock::utils::range_overlap;

#[derive(Default, Debug)]
pub struct SharedBuffer {
    /// `{ end key -> batch }`
    inner: BTreeMap<Vec<u8>, SharedBufferBatch>,
    size: u64,
}

impl SharedBuffer {
    pub fn write_batch(&mut self, batch: SharedBufferBatch) {
        self.inner
            .insert(batch.end_user_key().to_vec(), batch.clone());
        self.size += batch.size;
    }

    // Gets batches from shared buffer that overlap with the given key range.
    pub fn get_overlap_batches<R, B>(
        &self,
        key_range: &R,
        reversed_range: bool,
    ) -> Vec<SharedBufferBatch>
    where
        R: RangeBounds<B>,
        B: AsRef<[u8]>,
    {
        self.inner
            .range((
                if reversed_range {
                    key_range.end_bound().map(|b| b.as_ref().to_vec())
                } else {
                    key_range.start_bound().map(|b| b.as_ref().to_vec())
                },
                std::ops::Bound::Unbounded,
            ))
            .filter(|m| {
                range_overlap(
                    key_range,
                    m.1.start_user_key(),
                    m.1.end_user_key(),
                    reversed_range,
                )
            })
            .map(|entry| entry.1.clone())
            .collect()
    }

    pub fn delete_batch(&mut self, batch: SharedBufferBatch) -> Option<SharedBufferBatch> {
        self.inner.remove(batch.end_user_key())
    }

    pub fn size(&self) -> u64 {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use risingwave_hummock_sdk::key::{key_with_epoch, user_key};

    use super::*;
    use crate::hummock::iterator::test_utils::iterator_test_value_of;
    use crate::hummock::test_utils::default_config_for_test;
    use crate::hummock::value::HummockValue;

    async fn generate_and_write_batch(
        put_keys: &[Vec<u8>],
        delete_keys: &[Vec<u8>],
        epoch: u64,
        idx: &mut usize,
        shared_buffer_manager: &SharedBufferManager,
    ) -> SharedBufferBatch {
        let mut shared_buffer_items = Vec::new();
        for key in put_keys {
            shared_buffer_items.push((
                Bytes::from(key_with_epoch(key.clone(), epoch)),
                HummockValue::put(iterator_test_value_of(*idx).into()),
            ));
            *idx += 1;
        }
        for key in delete_keys {
            shared_buffer_items.push((
                Bytes::from(key_with_epoch(key.clone(), epoch)),
                HummockValue::delete(),
            ));
        }
        shared_buffer_items.sort_by(|l, r| user_key(&l.0).cmp(&r.0));
        shared_buffer_manager
            .write_batch(shared_buffer_items.clone(), epoch)
            .await
            .unwrap();
        SharedBufferBatch::new(shared_buffer_items, epoch)
    }

    #[tokio::test]
    async fn test_get_overlap_batches() {
        let shared_buffer_manager =
            SharedBufferManager::new(default_config_for_test().shared_buffer_capacity as usize);
        let mut keys = Vec::new();
        for i in 0..4 {
            keys.push(format!("key_test_{:05}", i).as_bytes().to_vec());
        }
        let large_key = format!("key_test_{:05}", 9).as_bytes().to_vec();
        let mut idx = 0;

        // Write a batch in epoch1
        let epoch1 = 1;
        let put_keys_in_epoch1 = &keys[..3];
        let shared_buffer_batch1 = generate_and_write_batch(
            put_keys_in_epoch1,
            &[],
            epoch1,
            &mut idx,
            &shared_buffer_manager,
        )
        .await;

        // Write a batch in epoch2, with key1 overlapping with epoch1 and key2 deleted.
        let epoch2 = epoch1 + 1;
        let put_keys_in_epoch2 = &[&keys[1..2], &keys[3..4]].concat();
        let shared_buffer_batch2 = generate_and_write_batch(
            put_keys_in_epoch2,
            &keys[2..3],
            epoch2,
            &mut idx,
            &shared_buffer_manager,
        )
        .await;

        // Case1: Get overlap batches in ..=epoch1
        for key in put_keys_in_epoch1 {
            // Single key
            let overlap_batches = shared_buffer_manager.get_overlap_batches(
                &(key.clone()..=key.clone()),
                ..=epoch1,
                false,
            );
            assert_eq!(overlap_batches.len(), 1);
            assert_eq!(overlap_batches[0], shared_buffer_batch1);

            // Forward key range
            let overlap_batches = shared_buffer_manager.get_overlap_batches(
                &(key.clone()..=keys[3].clone()),
                ..=epoch1,
                false,
            );
            assert_eq!(overlap_batches.len(), 1);
            assert_eq!(overlap_batches[0], shared_buffer_batch1);

            // Backward key range
            let overlap_batches = shared_buffer_manager.get_overlap_batches(
                &(keys[3].clone()..=key.clone()),
                ..=epoch1,
                true,
            );
            assert_eq!(overlap_batches.len(), 1);
            assert_eq!(overlap_batches[0], shared_buffer_batch1);
        }
        // Non-existent key
        let overlap_batches = shared_buffer_manager.get_overlap_batches(
            &(large_key.clone()..=large_key.clone()),
            ..=epoch1,
            false,
        );
        assert!(overlap_batches.is_empty());

        // Non-existent key range forward
        let overlap_batches = shared_buffer_manager.get_overlap_batches(
            &(keys[3].clone()..=large_key.clone()),
            ..=epoch1,
            false,
        );
        assert!(overlap_batches.is_empty());

        // Non-existent key range backward
        let overlap_batches = shared_buffer_manager.get_overlap_batches(
            &(large_key.clone()..=keys[3].clone()),
            ..=epoch1,
            true,
        );
        assert!(overlap_batches.is_empty());

        // Case2: Get overlap batches in ..=epoch2
        // Expected behavior:
        // - Case2.1: keys[0] -> batch1
        // - Case2.2: keys[1], keys[2] -> batch2, batch1
        // - Case2.3: keys[3] -> batch2

        // Case2.1
        let overlap_batches = shared_buffer_manager.get_overlap_batches(
            &(keys[0].clone()..=keys[0].clone()),
            ..=epoch2,
            false,
        );
        assert_eq!(overlap_batches.len(), 1);
        assert_eq!(overlap_batches[0], shared_buffer_batch1);

        // Case2.2
        for i in 1..=2 {
            // Single key
            let overlap_batches = shared_buffer_manager.get_overlap_batches(
                &(keys[i].clone()..=keys[i].clone()),
                ..=epoch2,
                false,
            );
            assert_eq!(overlap_batches.len(), 2);
            // Ordering matters. Fresher batches should be placed before older batches.
            assert_eq!(overlap_batches[0], shared_buffer_batch2);
            assert_eq!(overlap_batches[1], shared_buffer_batch1);

            // Forward key range
            let overlap_batches = shared_buffer_manager.get_overlap_batches(
                &(keys[0].clone()..=keys[i].clone()),
                ..=epoch2,
                false,
            );
            assert_eq!(overlap_batches.len(), 2);
            // Ordering matters. Fresher batches should be placed before older batches.
            assert_eq!(overlap_batches[0], shared_buffer_batch2);
            assert_eq!(overlap_batches[1], shared_buffer_batch1);

            // Backward key range
            let overlap_batches = shared_buffer_manager.get_overlap_batches(
                &(keys[i].clone()..=keys[0].clone()),
                ..=epoch2,
                true,
            );
            assert_eq!(overlap_batches.len(), 2);
            // Ordering matters. Fresher batches should be placed before older batches.
            assert_eq!(overlap_batches[0], shared_buffer_batch2);
            assert_eq!(overlap_batches[1], shared_buffer_batch1);
        }

        // Case2.3
        // Single key
        let overlap_batches = shared_buffer_manager.get_overlap_batches(
            &(keys[3].clone()..=keys[3].clone()),
            ..=epoch2,
            false,
        );
        assert_eq!(overlap_batches.len(), 1);
        assert_eq!(overlap_batches[0], shared_buffer_batch2);

        // Forward range
        let overlap_batches = shared_buffer_manager.get_overlap_batches(
            &(keys[3].clone()..=large_key.clone()),
            ..=epoch2,
            false,
        );
        assert_eq!(overlap_batches.len(), 1);
        assert_eq!(overlap_batches[0], shared_buffer_batch2);

        // Backward range
        let overlap_batches = shared_buffer_manager.get_overlap_batches(
            &(large_key.clone()..=keys[3].clone()),
            ..=epoch2,
            true,
        );
        assert_eq!(overlap_batches.len(), 1);
        assert_eq!(overlap_batches[0], shared_buffer_batch2);

        // Case3: Get overlap batches in epoch2..=epoch2
        for key in put_keys_in_epoch2 {
            // Single key
            let overlap_batches = shared_buffer_manager.get_overlap_batches(
                &(key.clone()..=key.clone()),
                epoch2..=epoch2,
                false,
            );
            assert_eq!(overlap_batches.len(), 1);
            assert_eq!(overlap_batches[0], shared_buffer_batch2);

            // Forward key range
            let overlap_batches = shared_buffer_manager.get_overlap_batches(
                &(key.clone()..=keys[3].clone()),
                epoch2..=epoch2,
                false,
            );
            assert_eq!(overlap_batches.len(), 1);
            assert_eq!(overlap_batches[0], shared_buffer_batch2);

            // Backward key range
            let overlap_batches = shared_buffer_manager.get_overlap_batches(
                &(keys[3].clone()..=key.clone()),
                epoch2..=epoch2,
                true,
            );
            assert_eq!(overlap_batches.len(), 1);
            assert_eq!(overlap_batches[0], shared_buffer_batch2);
        }
        // Non-existent key
        let overlap_batches = shared_buffer_manager.get_overlap_batches(
            &(keys[0].clone()..=keys[0].clone()),
            epoch2..=epoch2,
            false,
        );
        assert!(overlap_batches.is_empty());
    }
}
